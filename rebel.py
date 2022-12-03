
#In[1]: Dependencies and Pre-sets

import pandas as pd
import numpy as np
import json
import os
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Any, Union, List, Optional, Sequence

import re

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

import datasets
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

import transformers 
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    default_data_collator,
    set_seed,
)

import wandb
from pytorch_lightning.loggers.wandb import WandbLogger

cameo_to_penta = {
    "Make Public Statement" : "Make a statement",
    "Appeal" : "Verbal Cooperation",  	#Statement?
    "Express Intend to Cooperate" : "Verbal Cooperation",
    "Consult" : "Verbal Cooperation",
    "Engage In Diplomatic Cooperation" : "Verbal Cooperation",
    "Engage In Material Cooperation" : "Material Cooperation",
    "Provide Aid" : "Material Cooperation",
    "Yield" : "Material Cooperation",  #Verbal?
    "Investigate" : "Material Conflict",
    "Demand" : "Verbal Conflict",
    "Disapprove" : "Verbal Conflict",
    "Reject" : "Verbal Conflict",
    "Threaten" : "Verbal Conflict",
    "Exhibit Military Posture" : "Material Conflict",
    "Protest" : "Verbal Conflict", #Material?
    "Reduce Relations" : "Verbal Conflict",
    "Coerce" : "Material Conflict",
    "Assault" : "Material Conflict",
    "Fight" : "Material Conflict",
    "Engage in unconventional mass violence" : "Material Conflict"
}


#In[2]: Define dataset

#for pretrain data
data = pd.read_csv("soft_data/data/out_data/entail_articles_url_coref3.csv.done.csv", index_col = 0)
data = data.rename(columns = {"label":"triplets"})

#only if using coref3 or earlier
data["triplets"] = data.triplets.apply(lambda x: x.replace("Intend", "Express Intend to Cooperate"))

if sys.argv[1] == "pentacode":
    penta_map = []
    last_txt = ""
    for row in data.iterrows():
        trip_text = row[1]["triplets"]
        for key in cameo_to_penta.keys():
            if key in row[1]["triplets"]: trip_text = trip_text.replace(key, cameo_to_penta[key])
        penta_map.append([row[1]["text"], trip_text])
    data = pd.DataFrame(penta_map, columns = ["text","triplets"])
    print("initialized Pentacode data")

    penta_opts = ["Make a statement", "Verbal Cooperation", "Material Cooperation", "Verbal Conflict", "Material Conflict"]
    #extract the categorized relation for stratified split on pentacode
    relation = []
    for row in data.iterrows():
        sub_rels = []
        for opt in penta_opts:
            for count in range(row[1]["triplets"].count(opt)):
                sub_rels.append(opt)
        relation.append(sub_rels)
    data["relations"] = relation  

else: 
    #extract the categorized relation for stratified split on cameo codes
    relation = []
    for row in data.iterrows():
        sub_rels = []
        for cameo in cameo_to_penta.keys():
            for count in range(row[1]["label"].count(cameo)):
                sub_rels.append(cameo)
        relation.append(sub_rels)
    data["relations"] = relation      

#create stratified splits
mlb = MultiLabelBinarizer()
accept_MLB = mlb.fit_transform(data["relations"])

cols = [f"r{i}" for i in range(len(accept_MLB[0]))]
data = pd.concat([data, pd.DataFrame(accept_MLB, columns = cols)], axis=1)

#select indexes for train & val
splits = MultilabelStratifiedShuffleSplit(test_size=round(len(data.text) * 0.85), train_size= (len(data.text) - round(len(data.text) * 0.85)))
val_idx, train_idx = next(splits.split(data.text, data[cols]))

train = data.iloc[train_idx][["text", "triplets"]]
pre_split = data.iloc[val_idx]

#select indexes test & val
splits = MultilabelStratifiedShuffleSplit(test_size=round(len(pre_split.text) * 0.5), train_size= (len(pre_split.text) - round(len(pre_split.text) * 0.5)))
val_idx, test_idx = next(splits.split(pre_split.text, pre_split[cols]))

val = pre_split.iloc[val_idx][["text","triplets"]]
test = pre_split.iloc[test_idx][["text", "triplets"]]

print("train shape", train.shape)
print("val shape", val.shape)
print("test shape", test.shape)

data_dict = {
    "train": train, 
    "val": val, 
    "test": test
}

#In[3]: Define config

class conf:
    #general
    seed = 0
    gpus = 1 #more?
    ontology = sys.argv[1] #cameo or pentacode
    
    #input
    batch_size = 16
    max_length = 128
    ignore_pad_token_for_loss = True
    use_fast_tokenizer = True
    gradient_acc_steps = 1
    gradient_clip_value = 10.0
    load_workers = 50 #50 is 5/8 of plato
    masking = 1 #after how many epoch should new masking be applied, 0 = no masking

    #optimizer
    lr = 0.00005
    weight_decay = 0.01
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 0.00000001
    warm_up = 2 #num of epochs to warm_up on

    #training
    max_steps = 1200000
    samples_interval = 1000

    monitor_var  = "val_loss"
    monitor_var_mode = "min"
    # val_check_interval = 0.5
    # val_percent_check = 0.1

    model_name = "model1.pth"
    checkpoint_path = f"models/{model_name}"
    save_top_k = 1

    early_stopping = False
    patience = "5"

    length_penalty = 0
    no_repeat_ngram_size = 0
    num_beams = 3
    precision = 16
    amp_level = None

#In[5]: Pytorch Lightning DataModule

class GetData(pl.LightningDataModule):
    def __init__(self, conf: conf, tokenizer: AutoTokenizer, model: AutoModelForSeq2SeqLM):
        """init params from config"""
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.datasets = data_dict
        
        # Data collator
        label_pad_token_id = -100                       
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, self.model, label_pad_token_id=label_pad_token_id, padding = True)

    def preprocess_function(self, data):
        """tokenize, pad, truncate"""
        #split into input and labels
        inputs = data["text"]       
        outputs = data["triplets"]

        #process input
        model_inputs = self.tokenizer(inputs, max_length = conf.max_length, padding = True, truncation = True)

        #process labels
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(outputs, max_length = conf.max_length, padding = True, truncation = True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def apply_mask(self, data):        
        #masking entities in sentence
            print("applying mask ...")
            new = []
            for row in data.iterrows():
                split = re.split("<\w*>", row[1]["triplets"])[1:]   #first one is empty
                for i in range(int(len(split)/3)):                  #always pairs of 3
                    sub = split[i*3:i*3+3]
                    subj = sub[0]
                    obj = sub[1]             
                    if np.random.binomial(1, 0.15, 1) == 1:                  
                        tok = np.random.choice(sub).rstrip().lstrip()               
                        new.append([row[1]["text"].replace(tok, "<MASK>"), row[1]["triplets"].replace(tok, "<MASK>")])                    
                        break
                    else:
                        new.append([row[1]["text"], row[1]["triplets"]])
                    
            df = pd.DataFrame(new, columns = ["text", "triplets"])
            df = df.drop_duplicates("text")
            
            ds = datasets.Dataset.from_pandas(df)
            ds = ds.remove_columns(["__index_level_0__"])
            return ds 
        
    #apply the preprocessing and load data
    def train_dataloader(self, *args, **kwargs): 
        self.train_dataset = self.datasets["train"]
        if conf.masking != 0:
            self.train_dataset = self.apply_mask(self.train_dataset)
        self.train_dataset = self.train_dataset.map(self.preprocess_function, remove_columns = ["text", "triplets"], batched = True)
        return DataLoader(self.train_dataset, batch_size = conf.batch_size, collate_fn = self.data_collator, shuffle = True, num_workers= conf.load_workers)
    
    def val_dataloader(self, *args, **kwargs): 
        self.eval_dataset = self.datasets["val"]
        if conf.masking != 0:
            self.eval_dataset = self.apply_mask(self.eval_dataset)
        self.eval_dataset = self.eval_dataset.map(self.preprocess_function, remove_columns = ["text", "triplets"], batched = True)
        return DataLoader(self.eval_dataset, batch_size = conf.batch_size, collate_fn = self.data_collator, num_workers= conf.load_workers)

    def test_dataloader(self, *args, **kwargs): 
        self.test_dataset = self.datasets["test"]
        if conf.masking != 0:
            self.test_dataset = self.apply_mask(self.test_dataset)
        self.test_dataset = self.test_dataset.map(self.preprocess_function,  remove_columns = ["text", "triplets"], batched = True)
        return DataLoader(self.test_dataset, batch_size = conf.batch_size, collate_fn = self.data_collator, num_workers= conf.load_workers)

#In[6]: Pytorch Lightning Base module

class BaseModule(pl.LightningModule):

    def __init__(self, conf, config: AutoConfig, tokenizer: AutoTokenizer, model: AutoModelForSeq2SeqLM):
        super().__init__()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.ontology = conf.ontology
        #self.loss_fn = label_smoothed_nll_loss
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.num_beams = conf.num_beams

    def forward(self, inputs, labels, *args):
        ##### Check later if smooth labeled loss is better 
        outputs = self.model(**inputs, labels = labels, use_cache = False, return_dict = True, output_hidden_states = True)
        output_dict = {'loss': outputs['loss'], 'logits': outputs['logits']}
        return output_dict

    def training_step(self, batch: dict, batch_idx: int):
        ##### check later if labels = batch["labels"] also works
        labels = batch.pop("labels")
        labels_original = labels.clone()
        batch["decoder_input_ids"] = torch.where(labels != -100, labels, self.config.pad_token_id)
        labels = shift_tokens_left(labels, -100)

        forward_output = self.forward(batch, labels)
        self.log('loss', forward_output['loss'])

        batch["labels"] = labels_original

        #### ig i dont have this
        if 'loss_aux' in forward_output:
            self.log('loss_classifier', forward_output['loss_aux'])
            return forward_output['loss'] + forward_output['loss_aux']

        return forward_output['loss']

    def validation_step(self, batch: dict, batch_idx):
        #### pop maybe not needed?
        labels = batch.pop("labels")
        batch["decoder_input_ids"] = torch.where(labels != -100, labels, self.config.pad_token_id)
        labels = shift_tokens_left(labels, -100)
        
        with torch.no_grad():
            # compute loss on predict data
            forward_output = self.forward(batch, labels)

        forward_output['loss'] = forward_output['loss'].mean().detach()
        forward_output['logits'] = forward_output['logits'].detach()

        if labels.shape[-1] < conf.max_length:
            forward_output['labels'] = self._pad_tensors_to_max_len(labels, conf.max_length)
        else:
            forward_output['labels'] = labels

        metrics = {}
        metrics['val_loss'] = forward_output['loss']

        #### only 1? so why loop lmao
        for key in sorted(metrics.keys()):
            self.log(key, metrics[key])

        outputs = {}
        outputs['predictions'], outputs['labels'] = self.generate_triples(batch, labels)
        return outputs


    def test_step(self, batch, batch_idx):

        #### popping again
        labels = batch.pop("labels")
        batch["decoder_input_ids"] = torch.where(labels != -100, labels, self.config.pad_token_id)
        labels = shift_tokens_left(labels, -100)

        with torch.no_grad():
            # compute loss on predict data
            forward_output = self.forward(batch, labels)

        forward_output['loss'] = forward_output['loss'].mean().detach()

        forward_output['logits'] = forward_output['logits'].detach()

        if labels.shape[-1] < conf.max_length:
            forward_output['labels'] = self._pad_tensors_to_max_len(labels, conf.max_length)
        else:
            forward_output['labels'] = labels


        metrics = {}
        metrics['test_loss'] = forward_output['loss']

        #### dont i only have one metric anyways?
        for key in sorted(metrics.keys()):
            self.log(key, metrics[key], prog_bar=True)
        
        self.log("lr", optimizer.get_last_lr())

        #### what does this actually do? how does this change everything
        # if self.hparams.finetune:
        #     return {'predictions': self.forward_samples(batch, labels)}
        # else:

        outputs = {}
        outputs['predictions'], outputs['labels'] = self.generate_triples(batch, labels)
        return outputs



    def validation_epoch_end(self, output: dict):
        
        if self.ontology == "pentacode":
            relations = ["Make a statement", "Verbal Cooperation", "Material Cooperation", "Verbal Conflict", "Material Conflict"]
        else:
            relations = ["MakePublicStatement","Appeal","ExpressIntendToCooperate","Consult","EngageInDiplomaticCooperation","EngageInMaterialCooperation","ProvideAid","Yield","Investigate","Demand","Disapprove","Reject","Threaten","ExhibitMilitaryPosture","Protest","ReduceRelations","Coerce","Assault","Fight","EngageInUnconventialMassViolence"]
        
        scores, precision, recall, f1 = re_score([item for pred in output for item in pred['predictions']], [item for pred in output for item in pred['labels']], relations)
        self.log('val_prec_micro', precision)
        self.log('val_recall_micro', recall)
        self.log('val_F1_micro', f1)

    def test_epoch_end(self, output: dict):
        
        if self.ontology == "pentacode":
            relations = ["Make a statement", "Verbal Cooperation", "Material Cooperation", "Verbal Conflict", "Material Conflict"]
        else:
            relations = ["MakePublicStatement","Appeal","ExpressIntendToCooperate","Consult","EngageInDiplomaticCooperation","EngageInMaterialCooperation","ProvideAid","Yield","Investigate","Demand","Disapprove","Reject","Threaten","ExhibitMilitaryPosture","Protest","ReduceRelations","Coerce","Assault","Fight","EngageInUnconventialMassViolence"]
        
        scores, precision, recall, f1 = re_score([item for pred in output for item in pred['predictions']], [item for pred in output for item in pred['labels']], relations)
        self.log('test_prec_micro', precision)
        self.log('test_recall_micro', recall)
        self.log('test_F1_micro', f1)


    # additional functions called in main functions

    def generate_triples(self, batch, labels) -> None:

        generated_tokens = self.model.generate(
            batch["input_ids"].to(self.model.device),
            attention_mask=batch["attention_mask"].to(self.model.device),
            use_cache = True, max_length = conf.max_length, early_stopping = conf.early_stopping, length_penalty = conf.length_penalty, 
            no_repeat_ngram_size = conf.no_repeat_ngram_size, num_beams = conf.num_beams)

        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
        decoded_labels = self.tokenizer.batch_decode(torch.where(labels != -100, labels, self.config.pad_token_id), skip_special_tokens=False)

        return [extract_triplets(rel) for rel in decoded_preds], [extract_triplets(rel) for rel in decoded_labels]

    def _pad_tensors_to_max_len(self, tensor, max_length):
        # If PAD token is not defined at least EOS token has to be defined
        pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else self.config.eos_token_id

        if pad_token_id is None:
            raise ValueError(
                f"Make sure that either `config.pad_token_id` or `config.eos_token_id` is defined if tensor has to be padded to `max_length`={max_length}"
            )

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

    def configure_optimizers(self):

        ##### HUH
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": conf.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr = conf.lr, betas = (conf.beta1, conf.beta2), eps = conf.epsilon, weight_decay = conf.weight_decay)

        def lr_schedule(epoch):
            if epoch < conf.warm_up: lr_scale =  0.1
            else: lr_scale = 1 / (epoch**0.3)
            return lr_scale

        scheduler = lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_schedule
        )
        #scheduler = inverse_square_root(optimizer, num_warmup_steps= conf.warmup_steps)

        return [optimizer], [scheduler]


#In[7]: Helper functions

#from REBEL
class GenerateTextSamplesCallback(Callback):
    """
    PL Callback to generate triplets along training
    """

    def __init__(self, logging_batch_interval):
        """
        Args:
            logging_batch_interval: How frequently to inspect/potentially plot something
        """
        super().__init__()
        self.logging_batch_interval = logging_batch_interval

    def on_train_batch_end(self,trainer: Trainer,pl_module: LightningModule, outputs: Sequence, batch: Sequence, batch_idx: int) -> None:
        wandb_table = wandb.Table(columns=["Source", "Pred", "Gold"])
        # pl_module.logger.info("Executing translation callback")
        labels = batch.pop("labels")
        gen_kwargs = {
            "max_length": conf.max_length,
            "early_stopping": False,
            "no_repeat_ngram_size": 0,
            "num_beams": conf.num_beams
        }
        pl_module.eval()

        decoder_inputs = torch.roll(labels, 1, 1)[:,0:2]
        decoder_inputs[:, 0] = 0
        generated_tokens = pl_module.model.generate(
            batch["input_ids"].to(pl_module.model.device),
            attention_mask=batch["attention_mask"].to(pl_module.model.device),
            decoder_input_ids=decoder_inputs.to(pl_module.model.device),
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = pl_module._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])
        pl_module.train()
        decoded_preds = pl_module.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

        #If ignore pad token for loss == True 
            # Replace -100 in the labels as we can't decode them.
        labels = torch.where(labels != -100, labels, pl_module.tokenizer.pad_token_id)

        decoded_labels = pl_module.tokenizer.batch_decode(labels, skip_special_tokens=False)
        decoded_inputs = pl_module.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=False)

        # pl_module.logger.experiment.log_text('generated samples', '\n'.join(decoded_preds).replace('<pad>', ''))
        # pl_module.logger.experiment.log_text('original samples', '\n'.join(decoded_labels).replace('<pad>', ''))
        for source, translation, gold_output in zip(decoded_inputs, decoded_preds, decoded_labels):
            wandb_table.add_data(
                source.replace('<pad>', ''), translation.replace('<pad>', ''), gold_output.replace('<pad>', '')
            )
        pl_module.logger.experiment.log({"Triplets": wandb_table})

#from REBEL
def shift_tokens_left(input_ids: torch.Tensor, pad_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, :-1] = input_ids[:, 1:].clone()
    shifted_input_ids[:, -1] = pad_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."

    return shifted_input_ids

#from REBEL
def extract_triplets(text):
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
    return triplets

#from REBEL 
def score(key, prediction, verbose=False):
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation    = Counter()

    # Loop over the data to compute a score
    for row in range(len(prediction)):
        gold = key[row]
        guess = prediction[row]
         
        if gold == NO_RELATION and guess == NO_RELATION:
            pass
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1
        elif gold != NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    # Print verbose information
    if verbose:
        print("Per-relation statistics:")
        relations = gold_by_relation.keys()
        longest_relation = 0
        for relation in sorted(relations):
            longest_relation = max(len(relation), longest_relation)
        for relation in sorted(relations):
            # (compute the score)
            correct = correct_by_relation[relation]
            guessed = guessed_by_relation[relation]
            gold    = gold_by_relation[relation]
            prec = 1.0
            if guessed > 0:
                prec = float(correct) / float(guessed)
            recall = 0.0
            if gold > 0:
                recall = float(correct) / float(gold)
            f1 = 0.0
            if prec + recall > 0:
                f1 = 2.0 * prec * recall / (prec + recall)
            # (print the score)
            sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
            sys.stdout.write("  P: ")
            if prec < 0.1: sys.stdout.write(' ')
            if prec < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(prec))
            sys.stdout.write("  R: ")
            if recall < 0.1: sys.stdout.write(' ')
            if recall < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(recall))
            sys.stdout.write("  F1: ")
            if f1 < 0.1: sys.stdout.write(' ')
            if f1 < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(f1))
            sys.stdout.write("  #: %d" % gold)
            sys.stdout.write("\n")
        print("")

    # Print the aggregate score
    if verbose:
        print("Final Score:")
    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro   = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    print("Precision (micro): {:.3%}".format(prec_micro))
    print("   Recall (micro): {:.3%}".format(recall_micro))
    print("       F1 (micro): {:.3%}".format(f1_micro))
    return prec_micro, recall_micro, f1_micro

'''Adapted from: https://github.com/btaille/sincere/blob/6f5472c5aeaf7ef7765edf597ede48fdf1071712/code/utils/evaluation.py'''
def re_score(pred_relations, gt_relations, relation_types, mode="boundaries"):
    """Evaluate RE predictions
    Args:
        pred_relations (list) :  list of list of predicted relations (several relations in each sentence)
        gt_relations (list) :    list of list of ground truth relations
            rel = { "head": (start_idx (inclusive), end_idx (exclusive)),
                    "tail": (start_idx (inclusive), end_idx (exclusive)),
                    "head_type": ent_type,
                    "tail_type": ent_type,
                    "type": rel_type}
        vocab (Vocab) :         dataset vocabulary
        mode (str) :            in 'strict' or 'boundaries' """

    assert mode in ["strict", "boundaries"]

    if conf.ontology == "pentacode":
        relation_types = ["Make a statement", "Verbal Cooperation", "Material Cooperation", "Verbal Conflict", "Material Conflict"]
    else:
        relation_types = ["MakePublicStatement","Appeal","ExpressIntendToCooperate","Consult","EngageInDiplomaticCooperation","EngageInMaterialCooperation","ProvideAid","Yield","Investigate","Demand","Disapprove","Reject","Threaten","ExhibitMilitaryPosture","Protest","ReduceRelations","Coerce","Assault","Fight","EngageInUnconventialMassViolence"]
      
    # relation_types = [v for v in relation_types if not v == "None"]
    scores = {rel: {"tp": 0, "fp": 0, "fn": 0} for rel in relation_types + ["ALL"]}

    # Count GT relations and Predicted relations
    n_sents = len(gt_relations)
    n_rels = sum([len([rel for rel in sent]) for sent in gt_relations])
    n_found = sum([len([rel for rel in sent]) for sent in pred_relations])

    # Count TP, FP and FN per type
    for pred_sent, gt_sent in zip(pred_relations, gt_relations):
        for rel_type in relation_types:
            # strict mode takes argument types into account
            if mode == "strict":
                pred_rels = {(rel["head"], rel["head_type"], rel["tail"], rel["tail_type"]) for rel in pred_sent if
                             rel["type"] == rel_type}
                gt_rels = {(rel["head"], rel["head_type"], rel["tail"], rel["tail_type"]) for rel in gt_sent if
                           rel["type"] == rel_type}

            # boundaries mode only takes argument spans into account
            elif mode == "boundaries":
                pred_rels = {(rel["head"], rel["tail"]) for rel in pred_sent if rel["type"] == rel_type}
                gt_rels = {(rel["head"], rel["tail"]) for rel in gt_sent if rel["type"] == rel_type}

            scores[rel_type]["tp"] += len(pred_rels & gt_rels)
            scores[rel_type]["fp"] += len(pred_rels - gt_rels)
            scores[rel_type]["fn"] += len(gt_rels - pred_rels)

    # Compute per relation Precision / Recall / F1
    for rel_type in scores.keys():
        if scores[rel_type]["tp"]:
            scores[rel_type]["p"] = 100 * scores[rel_type]["tp"] / (scores[rel_type]["fp"] + scores[rel_type]["tp"])
            scores[rel_type]["r"] = 100 * scores[rel_type]["tp"] / (scores[rel_type]["fn"] + scores[rel_type]["tp"])
        else:
            scores[rel_type]["p"], scores[rel_type]["r"] = 0, 0

        if not scores[rel_type]["p"] + scores[rel_type]["r"] == 0:
            scores[rel_type]["f1"] = 2 * scores[rel_type]["p"] * scores[rel_type]["r"] / (
                    scores[rel_type]["p"] + scores[rel_type]["r"])
        else:
            scores[rel_type]["f1"] = 0

    # Compute micro F1 Scores
    tp = sum([scores[rel_type]["tp"] for rel_type in relation_types])
    fp = sum([scores[rel_type]["fp"] for rel_type in relation_types])
    fn = sum([scores[rel_type]["fn"] for rel_type in relation_types])

    if tp:
        precision = 100 * tp / (tp + fp)
        recall = 100 * tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

    else:
        precision, recall, f1 = 0, 0, 0

    scores["ALL"]["p"] = precision
    scores["ALL"]["r"] = recall
    scores["ALL"]["f1"] = f1
    scores["ALL"]["tp"] = tp
    scores["ALL"]["fp"] = fp
    scores["ALL"]["fn"] = fn

    # Compute Macro F1 Scores
    scores["ALL"]["Macro_f1"] = np.mean([scores[ent_type]["f1"] for ent_type in relation_types])
    scores["ALL"]["Macro_p"] = np.mean([scores[ent_type]["p"] for ent_type in relation_types])
    scores["ALL"]["Macro_r"] = np.mean([scores[ent_type]["r"] for ent_type in relation_types])

    print(f"RE Evaluation in *** {mode.upper()} *** mode")

    print(
        "processed {} sentences with {} relations; found: {} relations; correct: {}.".format(n_sents, n_rels, n_found,
                                                                                             tp))
    print(
        "\tALL\t TP: {};\tFP: {};\tFN: {}".format(
            scores["ALL"]["tp"],
            scores["ALL"]["fp"],
            scores["ALL"]["fn"]))
    print(
        "\t\t(m avg): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (micro)".format(
            precision,
            recall,
            f1))
    print(
        "\t\t(M avg): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (Macro)\n".format(
            scores["ALL"]["Macro_p"],
            scores["ALL"]["Macro_r"],
            scores["ALL"]["Macro_f1"]))

    for rel_type in relation_types:
        print("\t{}: \tTP: {};\tFP: {};\tFN: {};\tprecision: {:.2f};\trecall: {:.2f};\tf1: {:.2f};\t{}".format(
            rel_type,
            scores[rel_type]["tp"],
            scores[rel_type]["fp"],
            scores[rel_type]["fn"],
            scores[rel_type]["p"],
            scores[rel_type]["r"],
            scores[rel_type]["f1"],
            scores[rel_type]["tp"] +
            scores[rel_type][
                "fp"]))

    return scores, precision, recall, f1

#In[7]: Pytorch Trainer
def train(conf):
    pl.seed_everything(conf.seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained("Babelscape/rebel-large", use_fast = conf.use_fast_tokenizer,
        additional_special_tokens = ["<obj>", "<subj>", "<triplet>", "<head>", "</head>", "<tail>", "</tail>", "<MASK>"])
    config = transformers.AutoConfig.from_pretrained("Babelscape/rebel-large")
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")#, config = config)

    model.resize_token_embeddings(len(tokenizer))

    pl_data_module = GetData(conf, tokenizer, model)
    pl_module = BaseModule(conf, config, tokenizer, model)

    wandb_logger = WandbLogger(project = "project/finetune".split('/')[-1].replace('.py', ''), name = "finetune")

    callbacks_store = []

    if conf.early_stopping:
        callbacks_store.append(
            EarlyStopping(
                monitor=conf.monitor_var,
                mode=conf.monitor_var_mode,
                patience=conf.patience
            )
        )

    # callbacks_store.append(
    #     ModelCheckpoint(
    #         monitor=conf.monitor_var,
    #         # monitor=None,
    #         dirpath=f'models/{conf.model_name}',
    #         save_top_k=conf.save_top_k,
    #         verbose=True,
    #         save_last=True,
    #         mode=conf.monitor_var_mode
    #     )
    # )
    callbacks_store.append(GenerateTextSamplesCallback(conf.samples_interval))
    callbacks_store.append(LearningRateMonitor(logging_interval='step'))

    trainer = pl.Trainer(
        accelerator = "gpu",
        devices = conf.gpus,
        accumulate_grad_batches=conf.gradient_acc_steps,
        gradient_clip_val=conf.gradient_clip_value,
        #val_check_interval=conf.val_check_interval,
        max_epochs = 30,
        min_epochs = 5,
        callbacks=callbacks_store,
        max_steps=conf.max_steps,
        # max_steps=total_steps,
        reload_dataloaders_every_n_epochs = conf.masking, 
        precision=conf.precision,
        amp_level=conf.amp_level,
        logger=wandb_logger,
        #resume_from_checkpoint=conf.checkpoint_path,
        #limit_val_batches=conf.val_percent_check
    )

    trainer.fit(pl_module, datamodule=pl_data_module)

#In[8]: train model

train(conf)
#api key is fcfb005aa20d2c3af3389e0a2a6d58a829bfd2ee