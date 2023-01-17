# Thesis_RelationExtraction_PoliticsNews

The repository for my master thesis with the topic "Extracting political relations from news articles using transformers"

The leading research question is “How can political entities and the relations between them be extracted from news article data and classified using sequence-to-sequence transformer models?” 

~ Feel free to reach out for questions or comments


## Repository structure
'''bash
│   .gitignore
│   README.md
│   requirements.txt
│
├───data_src
│   │   Annot_to_sents.ipynb
│   │   DataAugmentation.py
│   │   DataAugmentationAnnotated.py
│   │   EDA_annotated.ipynb
│   │   EDA_unsupervised.ipynb
│   │   requirements.txt
│   │   
│   ├───annotated
│   ├───annotated_noaug
│   ├───conflibert,hu_et_al.2022
│   ├───raw
│   │   ├───in_label
│   │   └───out_label
│   │
│   └───unsupervised
│
├───docs
│   │   CAMEO Codebook.pdf
│   │   figures_thesis.pptx
│   │   Thesis_Werner_alpha_0.0.4.docx
│   ├───annotation_examples
│   └───generated_img
│
├───legacy
│
├───scrapers
│   │   scrape+coref.py
│   │
│   └───full_articles
│
└───src
    ├───test
    │       eval_conflibert.ipynb
    │       eval_rebel.ipynb
    │
    ├───train
    │       conflibert.py
    │       rebel.py
    │
    ├───tune
    │       conflibert_search.py
    │       conflibert_search_finetune.py
    │       rebel_search.py
    │       rebel_search_finetune.py
    │
    └───unsupervised_data
        │   auto_label_from_article.sh
        │   README.md
        │
        ├───data
        │   ├───out_data
        │   └───raw
        │
        └───src
            ├───add_labels
            │   │   check_entail.ipynb
            │   │   sent_to_triplet.py
            │   │
            │   └───dictionaries
            │
            └───preprocess
                    csv_to_xml.py
'''

## Script usage:

Both python scripts are running via command line. They are meant to be executed from the repository root.

First install the requirements using: pip install -r requirements.txt
---

Both training scripts take a multitude of parameters:

### for REBEL: python src/train/rebel.py ontology stage seed entity_hints augmentation run_name

### for ConfliBERT enc-dec: python src/train/conflibert.py ontology stage seed entity_hints augmentation run_name


valid inputs are:

- ontology = {cameo, pentacode}

- stage = {pretrain, finetune}

- seed = {0, 1, 2} (unless you create data for more seeds)

- entity_hints = {gold, spacy, none}

- augmentation = {aug, no_aug}

- run_name = any string (only used for identification)

## Training results

REBEL Training: https://wandb.ai/valentinwerner/REBEL?workspace=

REBEL Tuning pre-training: https://wandb.ai/valentinwerner/REBEL_hparams?workspace= 

REBEL Tuning fine-tune: https://wandb.ai/valentinwerner/REBEL_hparams_finetune?workspace=

ConfliBERT enc-dec Training: https://wandb.ai/valentinwerner/ConfliBERT?workspace=

ConfliBERT enc-dec Tuning pre-training: https://wandb.ai/valentinwerner/ConfliBERT_hparams?workspace=

ConfliBERT enc-dec Tuning fine-tune: https://wandb.ai/valentinwerner/ConfliBERT_hparams_finetune?workspace= 
