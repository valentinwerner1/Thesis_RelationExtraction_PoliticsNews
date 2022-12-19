# Thesis_RelationExtraction_PoliticsNews

The repository for my master thesis with the topic "Extracting political relations from news articles using transformers"

The leading research question is "Extracting political relations from news articles using transformers"


## Repository structure

- data_src                      #contains all relevant data used for training
|- raw                          #data files
|- EDA_annotated.ipynb  
|- EDA_soft_data.ipynb
|- requirements.txt             #requirements to run EDA
- docs                          #contains thesis, relevant files during writing the thesis
- legacy                        #contains outdated files, for lookup purposes 
- scrapers                      #contains scripts used to create datasets; incl. coreference-resolution
|- full_articles                #articles for soft_data
|- coref_isolated.py            #script to run only coreference resolution on a given text file
|- repesented_countries.json
|- scrape+coref.py              #script to run scraping and apply coreference resolution
|- requirements.txt             #requirements for scraping and coreference resolution
- soft_data                     #contains scripts for unsupervised annotation of triplets using syntax trees
|- data
    |- out_data                 #data annotated by script
    |- raw                      #data to be annotated by script
|- src
    |- add_labels       
        |- dictionaries         #PETRARCH dictionaries used to map to relations
        |- sent_to_triplet.py   #main script
    |- preprocess
        |- csv_to_xml.py        
|- auto_label_from_article.sh   #bash script to run full annotation
|- README.md                    #describes usage of bash script
|- requirements.txt             #requirements for bash script
- conflibert.py                 #script to train the seq2seq version of ConfliBERT
- rebel.py                      #script to train rebel
- requirements.txt              #requirements used to run training scripts

