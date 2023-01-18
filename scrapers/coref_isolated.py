# In[1]: Libraries

import pandas as pd

#Libraries for coref
import spacy
spacy.prefer_gpu()
from gensim import parsing
from crosslingual_coreference import Predictor
import en_core_web_sm

# In[2]: Run Coref

def run_coref(text):
    """applies coreference resolution to the new scraped articles"""
    #load spanbert model, as it is currently one of the state of the art models and achieved best performance on the data
    predictor = Predictor(language="en_core_web_sm", device = -1, model_name="spanbert")
    #apply coreference resolution
    text = str(text).replace(".", ". ")
    text = parsing.strip_multiple_whitespaces(text)
    coref_text = predictor.predict(text)["resolved_text"]   #resolved text already replaces resolution to input text
    
    return coref_text

df = pd.read_csv("full_articles/gdelt_reduced_scraped.csv", index_col=0)

new = []
for row in df.reset_index().iterrows():
    try:
        text = run_coref(row[1]["text"])    #get resolved text
        new.append([row[1]["paper"], row[1]["url"], text])    
    except:
        print("failed on ", row[1]["url"])  #catch excepts in case of fixing errors; indicates that scraped text isnt formatted properly

    if row[0] % 1000 == 0:
        #safe inbetween steps, as it takes long for many articles; allows checkpointing
        print(row[0])
        pd.DataFrame(new, columns = ["paper","url","text"]).to_csv("gdelt_coref_2.csv") 

pd.DataFrame(new, columns = ["paper","url","text"]).to_csv("gdelt_coref_2.csv")

# %%
