# Thesis_RelationExtraction_PoliticsNews

The repository for my master thesis with the topic "Extracting political relations from news articles using transformers"

The leading research question is “How can political entities and the relations between them be extracted from news article data and classified using sequence-to-sequence transformer models?” 

~ Feel free to reach out for questions or comments


## Repository structure

ADD file tree with bash tree


## Script usage:

Both python scripts are running via command line. They are meant to be executed from the repository root.

First install the requirements using: pip install -r requirements.txt


Both training scripts take a multitude of parameters:

for REBEL: python src/train/rebel.py ontology stage seed entity_hints augmentation run_name

for ConfliBERT enc-dec: python src/train/conflibert.py ontology stage seed entity_hints augmentation run_name


valid inputs are:

ontology -> cameo; pentacode

stage -> pretrain; finetune

seed -> 0; 1; 2 (unless you create data for more seeds)

entity_hints -> gold; spacy; none

augmentation -> aug; no_aug

run_name -> any string (only used for identification)

