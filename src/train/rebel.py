
#In[1]: Dependencies and Pre-sets

import pandas as pd
import numpy as np
import math
import json
import os
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Any, Union, List, Optional, Sequence

import re
from gensim.parsing.preprocessing import strip_multiple_whitespaces

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
    "Appeal" : "Make a statement", 
    "Express Intend to Cooperate" : "Verbal Cooperation",
    "Consult" : "Verbal Cooperation",
    "Engage In Diplomatic Cooperation" : "Verbal Cooperation",
    "Engage In Material Cooperation" : "Material Cooperation",
    "Provide Aid" : "Material Cooperation",
    "Yield" : "Material Cooperation", 
    "Investigate" : "Verbal Conflict",
    "Demand" : "Verbal Conflict",
    "Disapprove" : "Verbal Conflict",
    "Reject" : "Verbal Conflict",
    "Threaten" : "Verbal Conflict",
    "Exhibit Military Posture" : "Material Conflict",
    "Protest" : "Material Conflict", 
    "Reduce Relations" : "Verbal Conflict",
    "Coerce" : "Material Conflict",
    "Assault" : "Material Conflict",
    "Fight" : "Material Conflict",
    "Engage in unconventional mass violence" : "Material Conflict"
}

#Functions for one-time transforms
def get_spacy_hints(df, nlp):
    """extract noun chunks with relevant entity types and put entity markers in input text, induces a NER pipeline step"""
    new_text = []
    for row in df.iterrows():
        ents = []
        doc = nlp(row[1]["text"])
        new = row[1]["text"]
        for chunk in doc.noun_chunks:
            #check if any words of the chunk are relevant entities 
            if set([word.ent_type_ for word in chunk]) & set(["GPE", "NORP", "EVENTS", "FAC", "LAW", "ORG", "PERSON"]) != set():
                new = new.replace(chunk.text, "<ents> " + chunk.text + " </ents>")

        new_text.append([new, row[1]["triplets"]])
    print(new) #print one sample to validate
    return pd.DataFrame(new_text, columns = ["text","triplets"])

def get_gold_hints(df):
    """put entiy markers to all relevant entities (gold labeled) in text, simulates a perfect NER pipeline step"""
    new_text = []
    for row in df.iterrows():
        
        subj = re.findall("(?<=<triplet> ).*?(?= <subj>)", row[1]["triplets"])
        obj = re.findall("(?<=<subj> ).*?(?= <obj>)", row[1]["triplets"])

        ents = list(set(subj + obj))

        new = row[1]["text"]
        for ent in ents:
            new = new.replace(ent, "<ents> " + ent + " </ents>")

        new_text.append([new, row[1]["triplets"]])
    print(new) #print one sample to validate
    return pd.DataFrame(new_text, columns = ["text","triplets"])


#In[2]: Data Input

seed = sys.argv[3]

if sys.argv[2] == "pretrain":
    #for pre-train data
    train = pd.read_csv(f"data_src/unsupervised/train_{seed}.csv", index_col = 0)
    train = train.rename(columns = {"label":"triplets"})
    val = pd.read_csv(f"data_src/unsupervised/val_{seed}.csv", index_col = 0)
    val = val.rename(columns = {"label":"triplets"})
    test = pd.read_csv(f"data_src/unsupervised/test_{seed}.csv", index_col = 0)
    test = test.rename(columns = {"label":"triplets"})

elif sys.argv[2] == "finetune":
    #for fine-tune data
    if sys.argv[5] == "no_aug":
        train = pd.read_csv(f"data_src/annotated_noaug/train_{seed}.csv", index_col = 0)[["text","label"]]
        train = train.rename(columns = {"label":"triplets"})
        val = pd.read_csv(f"data_src/annotated_noaug/val_{seed}.csv", index_col = 0)[["text","label"]]
        val = val.rename(columns = {"label":"triplets"})
        test = pd.read_csv(f"data_src/annotated_noaug/test_{seed}.csv", index_col = 0)[["text","label"]]
        test = test.rename(columns = {"label":"triplets"})
        print("read data without augmentation")
    
    elif sys.argv[5] == "aug":
        train = pd.read_csv(f"data_src/annotated/new_train_aug_{seed}.csv", index_col = 0)
        train = train.rename(columns = {"label":"triplets"})
        val = pd.read_csv(f"data_src/annotated/new_val_aug_{seed}.csv", index_col = 0)
        val = val.rename(columns = {"label":"triplets"})
        test = pd.read_csv(f"data_src/annotated/new_test_aug_{seed}.csv", index_col = 0)
        test = test.rename(columns = {"label":"triplets"})
        print("read data with augmentation")

if sys.argv[1] == "pentacode":
    #map CAMEO labels to the respective Pentacode labels
    penta_map = []
    for row in train.iterrows():
        trip_text = row[1]["triplets"]
        for key in cameo_to_penta.keys():
            if key in row[1]["triplets"]: trip_text = trip_text.replace(key, cameo_to_penta[key])
        penta_map.append([row[1]["text"], trip_text])
    train = pd.DataFrame(penta_map, columns = ["text","triplets"])

    penta_map = []
    for row in val.iterrows():
        trip_text = row[1]["triplets"]
        for key in cameo_to_penta.keys():
            if key in row[1]["triplets"]: trip_text = trip_text.replace(key, cameo_to_penta[key])
        penta_map.append([row[1]["text"], trip_text])
    val = pd.DataFrame(penta_map, columns = ["text","triplets"])

    penta_map = []
    for row in test.iterrows():
        trip_text = row[1]["triplets"]
        for key in cameo_to_penta.keys():
            if key in row[1]["triplets"]: trip_text = trip_text.replace(key, cameo_to_penta[key])
        penta_map.append([row[1]["text"], trip_text])
    test = pd.DataFrame(penta_map, columns = ["text","triplets"])
    print("initialized Pentacode data")

if sys.argv[4] == "spacy":
    import spacy 
    nlp = spacy.load("en_core_web_trf")

    train = get_spacy_hints(train, nlp)
    val = get_spacy_hints(val, nlp)
    test = get_spacy_hints(test, nlp)
    print("initialized spacy hints")

elif sys.argv[4] == "gold":
    
    train = get_gold_hints(train)
    val = get_gold_hints(val)
    test = get_gold_hints(test)
    print("initialized gold hints")

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
    ontology = sys.argv[1] #cameo or pentacode
    gpu = 3

    #input
    if sys.argv[2] == "finetune":
        max_length = 128
        batch_size = 16
        gradient_acc_steps = 1 
        length_penalty = 1  #makes the model prefer to generate slightly longer sequences

        eps_loss = 0.2
        lr = 0.000025 
        lr_decay = 0.1
        masking_rate = 0.1
        weight_decay = 0.075

    elif sys.argv[2] == "pretrain":
        batch_size = 32
        max_length = 128
        gradient_acc_steps = 1
        length_penalty = 0

        eps_loss = 0.2
        lr = 0.00005
        lr_decay = 0.2
        masking_rate = 0.1
        weight_decay = 0.015

    ignore_pad_token_for_loss = True
    use_fast_tokenizer = True
    gradient_clip_value = 1
    masking = 1 #after how many epoch should new masking be applied, 0 = no masking

    #training
    monitor_var  = "val_F1_macro"
    monitor_var_mode = "max"

    #Callbacks (model saving; early stopping)
    samples_interval = 100
    model_name = sys.argv[6]
    checkpoint_path = f"test_models/{model_name}"
    save_top_k = 1
    early_stopping = True
    patience = 4
    save = True #save best model



#In[5]: Pytorch Lightning DataModule

class GetData(pl.LightningDataModule):
    def __init__(self, conf: conf, tokenizer: AutoTokenizer, model: AutoModelForSeq2SeqLM):
        """initialize params from config"""
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.conf = conf
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
        """applies the proposed entity masking to input and labels"""
        print("applying mask ...")
        new = []
        for row in data.iterrows():
            split = re.split("<\w*>", row[1]["triplets"])[1:]   #first one is empty
            for i in range(int(len(split)/3)):                  #always pairs of 3
                sub = split[i*3:i*3+3]           
                if np.random.binomial(1, self.conf.masking_rate, 1) == 1:                  
                    tok = np.random.choice(sub).rstrip().lstrip()               
                    new.append([row[1]["text"].replace(tok, "<MASK>"), row[1]["triplets"].replace(tok, "<MASK>")])                    
                    break
                else:
                    new.append([row[1]["text"], row[1]["triplets"]])
                
        df = pd.DataFrame(new, columns = ["text", "triplets"])
        df = df.drop_duplicates("text")
        
        ds = datasets.Dataset.from_pandas(df)
        try:
            ds = ds.remove_columns(["__index_level_0__"])
        except:
            pass
        return ds 
        
    #apply the preprocessing and load data
    def train_dataloader(self, *args, **kwargs): 
        """load training data, apply masking, tokenisation, truncation, padding"""
        self.train_dataset = self.datasets["train"]
        self.train_dataset = self.apply_mask(self.train_dataset)

        try:
            self.train_dataset = datasets.Dataset.from_pandas(self.train_dataset)
        except:
            pass

        try:
            self.train_dataset = self.train_dataset.remove_columns(["__index_level_0__"])
        except:
            pass

        self.train_dataset = self.train_dataset.map(self.preprocess_function, remove_columns = ["text", "triplets"], batched = True)
        return DataLoader(self.train_dataset, batch_size = self.conf.batch_size, collate_fn = self.data_collator, shuffle = True, num_workers= 50)
    
    def val_dataloader(self, *args, **kwargs): 
        self.val_dataset = self.datasets["val"]
        self.val_dataset = datasets.Dataset.from_pandas(self.val_dataset)

        try:
            self.val_dataset = self.val_dataset.remove_columns(["__index_level_0__"])
        except:
            pass

        self.val_dataset = self.val_dataset.map(self.preprocess_function, remove_columns = ["text", "triplets"], batched = True)
        return DataLoader(self.val_dataset, batch_size = self.conf.batch_size, collate_fn = self.data_collator, num_workers= 50)

    def test_dataloader(self, *args, **kwargs): 
        self.test_dataset = self.datasets["test"]
        self.test_dataset = datasets.Dataset.from_pandas(self.test_dataset)

        try:
            self.test_dataset = self.test_dataset.remove_columns(["__index_level_0__"])
        except:
            pass

        self.test_dataset = self.test_dataset.map(self.preprocess_function,  remove_columns = ["text", "triplets"], batched = True)
        return DataLoader(self.test_dataset, batch_size = self.conf.batch_size, collate_fn = self.data_collator, num_workers= 50)

#In[6]: Pytorch Lightning Base module

class BaseModule(pl.LightningModule):

    def __init__(self, conf, config: AutoConfig, tokenizer: AutoTokenizer, model: AutoModelForSeq2SeqLM):
        super().__init__()
        self.config = config
        self.conf = conf
        self.model = model
        self.tokenizer = tokenizer
        self.ontology = conf.ontology
        self.eps_loss = conf.eps_loss
        self.loss_fn = label_smoothed_nll_loss #torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.num_beams = 3

    def forward(self, inputs, labels, *args):
        """defines forward propagation step and calculates loss"""
        outputs = self.model(**inputs, use_cache=False, return_dict = True, output_hidden_states=True)
        logits = outputs['logits']
        lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        labels.masked_fill_(labels == -100, self.config.pad_token_id)
        loss, _ = self.loss_fn(lprobs, labels, self.eps_loss, ignore_index=self.config.pad_token_id)

        output_dict = {'loss': loss, 'logits':logits}
        return output_dict

    def training_step(self, batch: dict, batch_idx: int):
        """calls forward propagation"""
        labels = batch.pop("labels")
        labels_original = labels.clone()
        batch["decoder_input_ids"] = torch.where(labels != -100, labels, self.config.pad_token_id)
        labels = shift_tokens_left(labels, -100)

        forward_output = self.forward(batch, labels)
        self.log('loss', forward_output['loss'])

        batch["labels"] = labels_original

        forward_output['tr_loss'] = forward_output['loss'].mean().detach()
        if labels.shape[-1] < conf.max_length:
            forward_output['labels'] = self._pad_tensors_to_max_len(labels, conf.max_length)
        else:
            forward_output['labels'] = labels
        
        outputs = {}
        outputs['predictions'], outputs['labels'] = self.generate_triples(batch, labels)
        outputs["loss"] = forward_output['loss']
        return outputs

    def validation_step(self, batch: dict, batch_idx):
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

        for key in sorted(metrics.keys()):
            self.log(key, metrics[key])

        outputs = {}
        outputs['predictions'], outputs['labels'] = self.generate_triples(batch, labels)
        return outputs


    def test_step(self, batch, batch_idx):

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

        for key in sorted(metrics.keys()):
            self.log(key, metrics[key], prog_bar=True)

        outputs = {}
        outputs['predictions'], outputs['labels'] = self.generate_triples(batch, labels)
        return outputs

    def training_epoch_end(self, output: dict):
        """log accumulated scores at end of every epoch"""        
        print("\n\n train eval\n")
        
        scores, class_scores= re_score([item for pred in output for item in pred['predictions']], [item for pred in output for item in pred['labels']])

        self.log('train_prec_micro', scores["ALL"]["p"]) 
        self.log('train_recall_micro', scores["ALL"]["r"])
        self.log('train_F1_micro', scores["ALL"]["f1"])

        self.log("train_prec_macro", scores["ALL"]["Macro_p"])
        self.log("train_recall_macro", scores["ALL"]["Macro_r"])
        self.log("train_F1_macro", scores["ALL"]["Macro_f1"])

        self.log('train_prec_micro_class', class_scores["ALL"]["p"])
        self.log('train_recall_micro_class', class_scores["ALL"]["r"])
        self.log('train_F1_micro_class', class_scores["ALL"]["f1"])

        self.log("train_prec_macro_class", class_scores["ALL"]["Macro_p"])
        self.log("train_recall_macro_class", class_scores["ALL"]["Macro_r"])
        self.log("train_F1_macro_class", class_scores["ALL"]["Macro_f1"])

    def validation_epoch_end(self, output: dict):
        """log accumulated scores at end of every epoch"""
        print("\n\n validation eval\n")
       
        scores, class_scores = re_score([item for pred in output for item in pred['predictions']], [item for pred in output for item in pred['labels']])
        
        self.log('val_prec_micro', scores["ALL"]["p"]) 
        self.log('val_recall_micro', scores["ALL"]["r"])
        self.log('val_F1_micro', scores["ALL"]["f1"])

        self.log("val_prec_macro", scores["ALL"]["Macro_p"])
        self.log("val_recall_macro", scores["ALL"]["Macro_r"])
        self.log("val_F1_macro", scores["ALL"]["Macro_f1"])

        self.log('val_prec_micro_class', class_scores["ALL"]["p"])
        self.log('val_recall_micro_class', class_scores["ALL"]["r"])
        self.log('val_F1_micro_class', class_scores["ALL"]["f1"])

        self.log("val_prec_macro_class", class_scores["ALL"]["Macro_p"])
        self.log("val_recall_macro_class", class_scores["ALL"]["Macro_r"])
        self.log("val_F1_macro_class", class_scores["ALL"]["Macro_f1"])

    def test_epoch_end(self, output: dict):
        """log accumulated scores at end of every epoch"""       
        scores, class_scores = re_score([item for pred in output for item in pred['predictions']], [item for pred in output for item in pred['labels']])

        self.log('test_prec_micro', scores["ALL"]["p"]) 
        self.log('test_recall_micro', scores["ALL"]["r"])
        self.log('test_F1_micro', scores["ALL"]["f1"])

        self.log("test_prec_macro", scores["ALL"]["Macro_p"])
        self.log("test_recall_macro", scores["ALL"]["Macro_r"])
        self.log("test_F1_macro", scores["ALL"]["Macro_f1"])

        self.log('test_prec_micro_class', class_scores["ALL"]["p"])
        self.log('test_recall_micro_class', class_scores["ALL"]["r"])
        self.log('test_F1_micro_class', class_scores["ALL"]["f1"])

        self.log("test_prec_macro_class", class_scores["ALL"]["Macro_p"])
        self.log("test_recall_macro_class", class_scores["ALL"]["Macro_r"])
        self.log("test_F1_macro_class", class_scores["ALL"]["Macro_f1"])

    # additional functions called in main functions

    def generate_triples(self, batch, labels) -> None:
        """call on text generation"""
        generated_tokens = self.model.generate(
            batch["input_ids"].to(self.model.device),
            attention_mask=batch["attention_mask"].to(self.model.device),
            use_cache = True, max_length = conf.max_length, early_stopping = False, length_penalty = conf.length_penalty, 
            no_repeat_ngram_size = 0, num_beams = 3)

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
        """define optimizer, weight decay and lr schedule"""
        no_decay = ["bias", "LayerNorm.weight"]
        #no weight decay for the LayerNorm, rest is decayed
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.conf.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr = self.conf.lr, betas = (0.9, 0.999), eps = 0.00000001, weight_decay = self.conf.weight_decay)

        def lr_schedule(epoch):
            k = self.conf.lr_decay
            if epoch < 1: lr_scale =  0.1
            else: lr_scale = 1 * math.exp(-k*epoch)
            return lr_scale

        scheduler = lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_schedule
        )

        return [optimizer], [scheduler]


#In[7]: Helper functions

#FROM https://github.com/facebookresearch/fairseq/blob/main/fairseq/criterions/label_smoothed_cross_entropy.py
def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    """label smoothed nll loss, aiming at creating reducing overfitting and overconfidence"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

#from REBEL
class GenerateTextSamplesCallback(Callback):
    """PL Callback to generate triplets along training"""

    def __init__(self, logging_batch_interval):
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
            "num_beams": 3
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
    """extracts subject, object and relation from the generated text"""
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

#from REBEL; relation classification calculation added by me
'''Adapted from: https://github.com/btaille/sincere/blob/6f5472c5aeaf7ef7765edf597ede48fdf1071712/code/utils/evaluation.py'''
def re_score(pred_relations, gt_relations):
    """Evaluate RE predictions"""
    
    #define all relations
    if conf.ontology == "pentacode":
        relation_types = ["Make a statement", "Verbal Cooperation", "Material Cooperation", "Verbal Conflict", "Material Conflict"]
    else:
        if sys.argv[2] == "finetune":
            relation_types = ["Make Public Statement","Appeal","Express Intend to Cooperate","Consult","Engage In Diplomatic Cooperation",
            "Engage In Material Cooperation","Provide Aid","Yield","Investigate","Demand","Disapprove",
            "Reject","Exhibit Military Posture","Threaten","Protest","Reduce Relations","Coerce","Assault","Fight"]#,"Engage In Unconvential Mass Violence"]
        if sys.argv[2] == "pretrain":
            relation_types = ["Make Public Statement","Appeal","Express Intend to Cooperate","Consult","Engage In Diplomatic Cooperation",
            "Engage In Material Cooperation","Provide Aid","Yield","Investigate","Demand","Disapprove",
            #"Reject","Exhibit Military Posture",
            "Threaten","Protest","Reduce Relations","Coerce","Assault","Fight"]#,"Engage In Unconvential Mass Violence"]

    scores = {rel: {"tp": 0, "fp": 0, "fn": 0} for rel in relation_types + ["ALL"]}
    class_scores = {rel: {"tp": 0, "fp": 0, "fn": 0} for rel in relation_types + ["ALL"]}

    # Count GT relations and Predicted relations
    n_sents = len(gt_relations)
    n_rels = sum([len([rel for rel in sent]) for sent in gt_relations])
    n_found = sum([len([rel for rel in sent]) for sent in pred_relations])

    # Count TP, FP and FN per type
    for pred_sent, gt_sent in zip(pred_relations, gt_relations):
        for rel_type in relation_types:
            pred_rels = {(rel["head"], rel["tail"]) for rel in pred_sent if rel["type"] == rel_type}
            gt_rels = {(rel["head"], rel["tail"]) for rel in gt_sent if rel["type"] == rel_type}

            scores[rel_type]["tp"] += len(pred_rels & gt_rels)
            scores[rel_type]["fp"] += len(pred_rels - gt_rels)
            scores[rel_type]["fn"] += len(gt_rels - pred_rels)

            class_pred = [rel["type"] for rel in pred_sent if rel["type"] == rel_type]
            class_label = [rel["type"] for rel in gt_sent if rel["type"] == rel_type]

            class_scores[rel_type]["tp"] += min(len(class_pred), len(class_label))
            class_scores[rel_type]["fp"] += max(len(class_pred)-len(class_label),0)
            class_scores[rel_type]["fn"] += max(len(class_label)-len(class_pred),0)


    # Compute per triplet Precision / Recall / F1
    for rel_type in scores.keys():
        if scores[rel_type]["tp"]:
            scores[rel_type]["p"] = 100 * scores[rel_type]["tp"] / (scores[rel_type]["fp"] + scores[rel_type]["tp"])
            scores[rel_type]["r"] = 100 * scores[rel_type]["tp"] / (scores[rel_type]["fn"] + scores[rel_type]["tp"])
        else:
            scores[rel_type]["p"], scores[rel_type]["r"] = 0, 0

        if not scores[rel_type]["p"] + scores[rel_type]["r"] == 0:
            scores[rel_type]["f1"] = 2 * scores[rel_type]["p"] * scores[rel_type]["r"] / (scores[rel_type]["p"] + scores[rel_type]["r"])
        else:
            scores[rel_type]["f1"] = 0

    # Compute per class Precision / Recall / F1
    for rel_type in scores.keys():
        if class_scores[rel_type]["tp"]:
            class_scores[rel_type]["p"] = 100 * class_scores[rel_type]["tp"] / (class_scores[rel_type]["fp"] + class_scores[rel_type]["tp"])
            class_scores[rel_type]["r"] = 100 * class_scores[rel_type]["tp"] / (class_scores[rel_type]["fn"] + class_scores[rel_type]["tp"])
        else:
            class_scores[rel_type]["p"], class_scores[rel_type]["r"] = 0, 0

        if not class_scores[rel_type]["p"] + class_scores[rel_type]["r"] == 0:
            class_scores[rel_type]["f1"] = 2 * class_scores[rel_type]["p"] * class_scores[rel_type]["r"] / (class_scores[rel_type]["p"] + class_scores[rel_type]["r"])
        else:
            class_scores[rel_type]["f1"] = 0

    # Compute micro F1 Scores, relations
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

    # Compute micro F1 Scores, classes
    class_tp = sum([class_scores[rel_type]["tp"] for rel_type in relation_types])
    class_fp = sum([class_scores[rel_type]["fp"] for rel_type in relation_types])
    class_fn = sum([class_scores[rel_type]["fn"] for rel_type in relation_types])

    if class_tp:
        class_precision = 100 * class_tp / (class_tp + class_fp)
        class_recall = 100 * class_tp / (class_tp + class_fn)
        class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall)

    else:
        class_precision, class_recall, class_f1 = 0, 0, 0

    class_scores["ALL"]["p"] = class_precision
    class_scores["ALL"]["r"] = class_recall
    class_scores["ALL"]["f1"] = class_f1
    class_scores["ALL"]["tp"] = class_tp
    class_scores["ALL"]["fp"] = class_fp
    class_scores["ALL"]["fn"] = class_fn

    # Compute Macro F1 Scores, relations
    scores["ALL"]["Macro_f1"] = np.mean([scores[rel_type]["f1"] for rel_type in relation_types])
    scores["ALL"]["Macro_p"] = np.mean([scores[rel_type]["p"] for rel_type in relation_types])
    scores["ALL"]["Macro_r"] = np.mean([scores[rel_type]["r"] for rel_type in relation_types])

    print(f"Full Triplet Evaluation")
    print(
        "processed {} sentences with {} relations; found: {} relations; correct: {}.".format(n_sents, n_rels, n_found,tp))
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
            scores[rel_type]["fp"]))

    # Compute Macro F1 Scores, relations
    class_scores["ALL"]["Macro_f1"] = np.mean([class_scores[rel_type]["f1"] for rel_type in relation_types])
    class_scores["ALL"]["Macro_p"] = np.mean([class_scores[rel_type]["p"] for rel_type in relation_types])
    class_scores["ALL"]["Macro_r"] = np.mean([class_scores[rel_type]["r"] for rel_type in relation_types])

    print(f"Relation Classification Evaluation")

    print(
        "processed {} sentences with {} relations; found: {} relations; correct: {}.".format(n_sents, n_rels, n_found,class_tp))
    print(
        "\tALL\t TP: {};\tFP: {};\tFN: {}".format(
            class_scores["ALL"]["tp"],
            class_scores["ALL"]["fp"],
            class_scores["ALL"]["fn"]))
    print(
        "\t\t(m avg): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (micro)".format(
            class_precision,
            class_recall,
            class_f1))
    print(
        "\t\t(M avg): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (Macro)\n".format(
            class_scores["ALL"]["Macro_p"],
            class_scores["ALL"]["Macro_r"],
            class_scores["ALL"]["Macro_f1"]))

    for rel_type in relation_types:
        print("\t{}: \tTP: {};\tFP: {};\tFN: {};\tprecision: {:.2f};\trecall: {:.2f};\tf1: {:.2f};\t{}".format(
            rel_type,
            class_scores[rel_type]["tp"],
            class_scores[rel_type]["fp"],
            class_scores[rel_type]["fn"],
            class_scores[rel_type]["p"],
            class_scores[rel_type]["r"],
            class_scores[rel_type]["f1"],
            class_scores[rel_type]["tp"] +
            class_scores[rel_type]["fp"]))

    return scores, class_scores

#In[7]: Pytorch Trainer
def train(conf):
    #set seed for everything for reproducability
    pl.seed_everything(int(seed))

    print(f" batch_size: {conf.batch_size * conf.gradient_acc_steps} \n learning rate: {conf.lr} \n learning rate decay: {conf.lr_decay} \n weight decay: {conf.weight_decay} \n epsilon loss: {conf.eps_loss} \n gradient clipping: {conf.gradient_clip_value} \n accumulation steps: {conf.gradient_acc_steps}")

    #note that all reported results except for ablation were trained without the "<ents>" tokens and results may slightly differ if these tokens are initiliased
    if sys.argv[4] == "gold" or sys.argv[4] == "spacy":
        add_tokens = ["<obj>", "<subj>", "<triplet>", "<head>", "</head>", "<tail>", "</tail>", "<ents>", "</ents>", "<MASK>"]
    else:
        add_tokens = ["<obj>", "<subj>", "<triplet>", "<head>", "</head>", "<tail>", "</tail>", "<MASK>"]

    #tokenizer, config, and model loaded from Huggingface
    tokenizer = transformers.AutoTokenizer.from_pretrained("Babelscape/rebel-large", use_fast = True,
        additional_special_tokens = add_tokens)
    config = transformers.AutoConfig.from_pretrained("Babelscape/rebel-large")
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large", config = config)
    
    #add new embeddings for newly added tokens
    model.resize_token_embeddings(len(tokenizer))

    # #load pre-trained checkpoint (models are not included with code for size reasons)
    # if sys.argv[2] == "finetune":
    #     if sys.argv[1] == "pentacode":
    #         #the map possibly needs to be adjusted or removed - if all GPUs are free, just remove it 
    #         ckpt = torch.load("models/REBEL_pre-train_0_Penta.pth/epoch=19-step=10040.ckpt", map_location={'cuda:3':'cuda:0'})
    #     elif sys.argv[1] == "cameo":
    #         if sys.argv[4] == "gold" or sys.argv[4] == "spacy":
    #             ckpt = torch.load("models/REBEL_pre-train_ents/epoch=19-step=10040.ckpt", map_location={"cuda:1":"cuda:0"})
    #         else:
    #             ckpt = torch.load("models/REBEL_pre-train_0_CAMEO.pth/epoch=19-step=10040.ckpt", map_location={'cuda:0':'cuda:0'})

    #     else: 
    #         print("\nno correct ontology given\n")

    #     #the checkpoint adds the prefix "model." to layers, which we need to remove
    #     state_dict = {}
    #     for key in ckpt["state_dict"].keys():
    #         state_dict[key[6:]] = ckpt["state_dict"][key]

    #     model.load_state_dict(state_dict)
    #     print("checkpoint loaded")

    pl_data_module = GetData(conf, tokenizer, model)
    pl_module = BaseModule(conf, config, tokenizer, model)

    wandb_logger = WandbLogger(project = "project/REBEL".split('/')[-1].replace('.py', ''), name = f"{sys.argv[1]},{sys.argv[2]},seed{sys.argv[3]};{sys.argv[6]}", log_model = False)

    callbacks_store = []

    callbacks_store.append(GenerateTextSamplesCallback(100))
    callbacks_store.append(LearningRateMonitor(logging_interval='epoch'))

    if conf.early_stopping:
        callbacks_store.append(
            EarlyStopping(
                monitor=conf.monitor_var,
                mode=conf.monitor_var_mode,
                patience=conf.patience
            )
        )

    if conf.save:
        callbacks_store.append(
            ModelCheckpoint(
                monitor=conf.monitor_var,
                dirpath=conf.checkpoint_path,
                save_top_k=conf.save_top_k,
                verbose=True,
                save_last=False,
                mode=conf.monitor_var_mode
            )
        )

    trainer = pl.Trainer(
        accelerator = "gpu",
        devices = [2],
        accumulate_grad_batches=conf.gradient_acc_steps,
        gradient_clip_val=conf.gradient_clip_value,
        max_epochs = 25,
        min_epochs = 5,
        callbacks=callbacks_store,
        reload_dataloaders_every_n_epochs = 1, 
        precision=16,
        amp_level=None,
        logger=wandb_logger,
    )

    trainer.fit(pl_module, datamodule=pl_data_module)

    #evaluate on highest performing checkpoint
    print("finished training")
    ckpt = torch.load(f"test_models/{sys.argv[6]}/{os.listdir(f'test_models/{sys.argv[6]}')[0]}")   

    #the checkpoint adds the prefix "model." to layers, which we need to remove
    state_dict = {}
    for key in ckpt["state_dict"].keys():
        state_dict[key[6:]] = ckpt["state_dict"][key]

    model.load_state_dict(state_dict)
    print("reloaded best model")

    trainer.test(pl_module, datamodule=pl_data_module)


#In[8]: train model

train(conf)
