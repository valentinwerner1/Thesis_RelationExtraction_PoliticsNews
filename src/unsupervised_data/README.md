## Credits where Credits is due

The idea behind this code is a variation of https://github.com/eventdata/UDPetrarch-for-Python3 which was heavily alligned for the use case of this project.

None of the code is copied directly from the repository mentioned above, the only files that are identical are dictionary files.


## Annotation

Takes a csv file of full length articles, turns it to an xml file to split into sentences using CoreNLP. The separate sentences are then classified into part-of-speech-tokens using spacy and subj-verb-obj triplets are built. If the verb is existing in the CAMEO dictionairy, the respective category is assigned.

All dictionaries are not created by me, but by the researchers behind CAMEO and PETRARCH.

## Usage

First, you will need to install Stanford CoreNLP (http://stanfordnlp.github.io/CoreNLP/download.html) for sentence splitting 

Next, you will need to replace paths in the auto_label_from_article.sh script to your respective paths.

When using the script the first time, you should install the requirements, by uncommenting the first two script lines for preparing the environment.



