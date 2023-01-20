##Objective of the script: Split data into train/val/test for a given seed and augment train data to reduce class imbalance durin

#In[1]: Imports and Functions

import pandas as pd
import numpy as np
import random
import re
from gensim.parsing.preprocessing import strip_multiple_whitespaces

import seaborn as sns
import matplotlib.pyplot as plt 

import spacy
spacy.prefer_gpu(2) #maybe us en_core_web_lg instead of trf if no gpu
from spacy.symbols import nsubj, dobj, pobj, iobj, neg, xcomp, ccomp, VERB, AUX
nlp = spacy.load('en_core_web_trf')

from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from transformers import pipeline
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

from sklearn.metrics.pairwise import cosine_similarity

import sys
seed = sys.argv[1]

import os
os.chdir("..")

print("set the seed for this run to ", seed)

def read_lines(inputparsed):    
    """takes input from CoreNLP sentence parsed file and returns sentences"""
    #parse all lines from CoreNLP sentence split
    parsed = open(inputparsed, encoding = "utf-8")
    parsedfile = parsed.readlines()
    parsedlines = []

    #Only keep those lines which have Sentence #n in the line before
    for idx, text in enumerate(parsedfile):
        if text.startswith("Sentence #"):
            parsedlines.append(parsedfile[idx+1].replace('\n','').strip())
    
    return parsedlines

def gen_poss(line, verb_match, pre_dict):
    """generates all possibilities of patterns that a multi word line implies,
    by extracting partial patterns and resolving placeholder words"""
    poss = []

    #replace special tokens in text that are clear at this point
    line = line.replace("*", verb_match)
    line = line.replace("- ", "")
    line = line.replace("+","")
    line = line.replace("%","")
    line = line.replace("^","")
    line = line.replace("$","")

    #split line by possibility indicators and code (always ends possibility)
    #example.: "- $ * (P ON KILLING (P OF + [010] #  COMMENT <ELH 07 May 2008>"
    poss_split = re.split("\(P |\[.*]",line) 

    if len(poss_split) > 2: #2 is if no (P in the line
        #only combining the first (P, as they share the same code 
        #and the longer version will never be contained in a text if the shorter isnt
        poss.append(strip_multiple_whitespaces(" ".join(poss_split[:2])).lower().rstrip().lstrip())
    else: 
        poss.append(strip_multiple_whitespaces(poss_split[0].lower().rstrip().lstrip()))

    cleaned = []
    for text in list(set(poss)):
        c = 0
        for tag in list(pre_dict.keys()):
            if tag in text:
                for replacement in pre_dict[tag]:
                    cleaned.append(text.replace(tag, replacement))
                    c += 1
        if c == 0:
            cleaned.append(text)

    return cleaned
    

def verb_code_dict(pico_path, verb_path):
    """reads coding ontology and verb lists, 
    directly matches verbs to their CAMEO codes and returns this verbs:codes dictionairy.
    verb with codes that cannot be read are printed out as full line of the file"""
    #read PETRARCH Internal Coding Ontology (= pico)
    pico_path = os.path.join(os.getcwd(), pico_path)
    pico_file = open(pico_path, 'r')
    pico_lines = pico_file.readlines()

    #get all 20 codes with their respective code
    main_codes = {}                             #we run one iteration for all the main codes, only main codes contain relation name
    for line in pico_lines:
        line = line.split('#')
        if line[0] == "" or line[0] == "\n":    #only intro comments and empty lines
            continue
        else: 
            code_split = line[0].split(":")     #splits into CAMEO code and related hex
            if len(line) > 1 and code_split[0][2] == "0":      #only main categories have 0 in 3rd idx, [cat_num 0] -> [010]
                main_codes[code_split[0][:2]] = line[-1].replace("\n","")
    
    #map code to code we want to use in the training
    map_codes = {"DiplomaticCoop" : "Engage In Diplomatic Cooperation", 
                "MaterialCoop" : "Engage In Material Cooperation",
                "ProvideAid" : "Provide Aid",
                "Intend" : "Express Intend to Cooperate",
                "Exhibit Force Posture": "Exhibit Military Posture",
                "Use Unconventional Mass Violence" : "Engage In Unconventional Mass Violence"}
    main_codes = {k: (map_codes[v] if v in map_codes else v) for k, v in main_codes.items()}
    
    #read single word patterns and match their code to the relation extracted in main_codes
    verb_path = os.path.join(os.getcwd(), verb_path)
    verb_file = open(verb_path, 'r')
    verb_lines = verb_file.readlines()
    
    verb_dict = {}
    for line in verb_lines:
        if line[0] == "#":
            continue
        elif line.startswith("---"):    #main verbs have a lead code, which is applied to all very in the section
                                        #unless a separate code is specified for a specific verb in section
            try: cur_main_code = re.split("\[|\]|---", line)[2].replace(":","")[:2]  #we only need main codes which are first two numbers
                                                                                #sometimes code starts with ":", e.g.: ---  OFFEND   [:110]  ---
                                                                                #we just remove those to get the main code
            except:                     #depending on chosen verb dictionairy, there may be main verbs without lead codes
                print("couldn't finde code in: ", line.replace("\n","")) 
                cur_main_code == "--"
            if cur_main_code == "": cur_main_code = "--"
        elif line == "\n":              #skip empty lines
            continue
        elif line[0] == "-" or line[0] == "~" or line[0] == "+" or line[0] == "&": #removes all special structures we cannot use
            continue
        else:
            if len(re.split("\[|\]", line)) > 1:    #verbs with their own code, e.g.: AFFIRM [051] 
                code = re.split("\[|\]", line)[1].replace(":","")[:2]
                if code != "--":
                    if "{" in line:         #conjugated verbs, e.g. "APPLY {APPLYING APPLIED APPLIES } [020]"
                        line_s = re.split("\{|\}", line)    #split at { and }
                        verb_dict[line_s[0].lower().rstrip().lstrip()] = main_codes[code] 
                        for word in line_s[1].split():
                            verb_dict[word.lower().rstrip().lstrip()] = main_codes[code]
                    else:
                        word = re.split("\[|\]", line)[0]
                        verb_dict[word.lower().rstrip().lstrip()] = main_codes[code]
            else:
                if cur_main_code != "--":
                    if "{" in line:         #e.g. "HURRY {HURRIES HURRYING HURRIED }" 
                        line_s = re.split("\{|\}", line)    #split at { and }
                        verb_dict[line_s[0].lower().rstrip().lstrip()] = main_codes[cur_main_code]
                        for word in line_s[1].split():
                            verb_dict[word.lower().rstrip().lstrip()] = main_codes[cur_main_code]
                    else:                   #only single words with sometimes comments, e.g.: CENSURE  # JON 5/17/95
                        word = line.split("#")[0].rstrip()    #gets part before "#", removes all whitespaces to the right
                        verb_dict[word.lower().rstrip().lstrip()] = main_codes[cur_main_code]

    #read multi word patterns and create a dictionary for their code

    #get filler words that occur in multi word patterns
    verb_file = open(verb_path, 'r')
    verb_lines = verb_file.readlines()

    pre_dict = {}
    filter_list = []
    for line in verb_lines:
        if line.startswith("&"):
            cur_filter = line.rstrip()
        elif line.startswith("\n") and "cur_filter" in locals():
            pre_dict[cur_filter.lower()] = filter_list
            cur_filter = ""
            filter_list = []
        elif line.startswith("+") and cur_filter != "":
            filter_list.append(line.rstrip()[1:].replace("_", "").lower())
    del pre_dict[""]

    #generate dictionaries for multi word patterns
    verb_file = open(verb_path, 'r')
    verb_lines = verb_file.readlines()
    
    spec_dict = {}
    spec_code = {}

    count = 0
    for line in verb_lines:
        if line.startswith("- "):
            #get main verb as dict key
            try: 
                verb_match = re.search("# *\w+", line).group()
                verb_match = re.search("\w+", verb_match).group()
                verb_match = verb_match.replace("_", " ").lower()
            except: 
                count += 1

            #get code for line
            try:
                code = re.search("\[.*]", line).group()[1:3]
                if code != "--":
                    #get all possibility that the line indicates
                    poss = gen_poss(line, verb_match, pre_dict)
                    for pattern in poss:
                        spec_code[pattern] = main_codes[code]
                        if verb_match in spec_dict.keys():
                            spec_dict[verb_match].append(pattern)
                        else:
                            spec_dict[verb_match] = [pattern]
            except:
                count += 1

    print(f"{count} patterns could not be loaded")        

    return verb_dict, spec_dict, spec_code

def draw_new_ent(ent, taken_ents):
    ent_type = [word.ent_type_ for word in nlp(ent)]
    if "PERSON" in ent_type: 
        remainder = list(set(ent_dict_deduped["PERSON"]).difference(set(taken_ents)))
        new_ent = np.random.choice(ent_dict_deduped["PERSON"])
    elif "GPE" in ent_type: 
        remainder = list(set(ent_dict_deduped["GPE"]).difference(set(taken_ents)))
        new_ent = np.random.choice(ent_dict_deduped["GPE"])
    elif "NORP" in ent_type: 
        remainder = list(set(ent_dict_deduped["NORP"]).difference(set(taken_ents)))
        new_ent = np.random.choice(ent_dict_deduped["NORP"])
    elif "EVENTS" in ent_type:
        remainder = list(set(ent_dict_deduped["EVENTS"]).difference(set(taken_ents)))
        new_ent = np.random.choice(ent_dict_deduped["EVENTS"])
    elif "FAC" in ent_type: 
        remainder = list(set(ent_dict_deduped["FAC"]).difference(set(taken_ents)))
        new_ent = np.random.choice(ent_dict_deduped["FAC"])
    elif "LAW" in ent_type: 
        remainder = list(set(ent_dict_deduped["LAW"]).difference(set(taken_ents)))
        new_ent = np.random.choice(ent_dict_deduped["LAW"])
    elif "ORG" in ent_type: 
        remainder = list(set(ent_dict_deduped["ORG"]).difference(set(taken_ents)))
        new_ent = np.random.choice(ent_dict_deduped["ORG"])
    
    try: 
        if new_ent: return new_ent
    except: 
        #if no substitute can be found for one entity, the other entity can still be augmented
        return ent


#In[2]: 

# data = pd.read_csv("data_src/annotated_data_noaug.csv", index_col = 0)
# data.columns = ["doc_id","text","label","rel_count","len","new_len","in_len","relations","rel_len"]

#df= pd.read_csv("data_src/unsupervised_data_preprocessed.csv", index_col = 0)
#the preprocessed data already removed inputs longer than 500 symbols and downsampled Consult & Make public statement
#see EDA_unsupervised.ipynb for more info

verb_dict, spec_dict, spec_code = verb_code_dict("soft_data/src/add_labels/dictionaries/PETR.Internal.Coding.Ontology.txt", "soft_data/src/add_labels/dictionaries/newdict.txt")

full = pd.read_csv("data_src/annotated/sent_full.csv")
cameo = pd.read_csv("data_src/annotated/CAMEO.csv")[["text","triplets"]]
cameo.columns = ["text","label"]

cameo["label"] = cameo.label.apply(lambda x: x.replace("Express Intent to Cooperate","Express Intend to Cooperate"))

relation = []
for row in cameo.iterrows():
    rel_iter = row[1]["label"] + " <triplet>"
    all_rels = re.findall("(?<=<obj> ).*?(?= <triplet>)", rel_iter)
    relation.append(all_rels)
cameo["relations"] = relation

cameo["mps"] = cameo.relations.apply(lambda x: x.count("Make Public Statement"))
cameo["len"] = cameo.relations.apply(lambda x: len(x))

mps = []
for row in cameo.iterrows():
    mps.append(row[1]["mps"] / row[1]["len"])
cameo["MPS%"] = mps

mps = cameo[(cameo["MPS%"] > 0.6) & (cameo.mps > 1)]

cameo["unq"] = cameo.relations.apply(lambda x: len(list(set(x))))

new_sub=[]
for row in mps.iterrows():

    try:
        ex = row[1]["text"]

        rm = ex[ex.rindex(',')+1:]
        label = row[1]["label"]

        remainder = ex[:-(len(ex[ex.rindex(','):]))]

        rels = []
        split = re.split("<\w*>", label)[1:] #first one is empty
        for i in range(int(len(split)/3)): #always pairs of 3
            sub = split[i*3:i*3+3]
            rels.append([sub[0].lstrip().rstrip(), sub[1].lstrip().rstrip(), sub[2].lstrip().rstrip()])
        
        keep = []
        for rel in rels:
            if rel[2] == "Make Public Statement":

                if rel[0] in rm or rel[1] in rm:
                    #print(rel)
                    continue
                else: 
                    keep.append(rel)
            else: keep.append(rel)
        
        new_rel = ""
        for rel in keep:
            new_rel += f"<triplet> {rel[0]} <subj> {rel[1]} <obj> {rel[2]} "
        new_sub.append([remainder,new_rel.rstrip()])

    except:
        new_sub.append([row[1]["text"], row[1]["label"]])

cameo = cameo.drop(index = mps.index)
new = pd.DataFrame(new_sub, columns = ["text","label"])
cameo = pd.concat([cameo, new]).reset_index().drop(columns = ["index"])[["text","label"]]
relation = []
for row in cameo.iterrows():
    rel_iter = row[1]["label"] + " <triplet>"
    all_rels = re.findall("(?<=<obj> ).*?(?= <triplet>)", rel_iter)
    relation.append(all_rels)
cameo["relations"] = relation

cameo["len"] = cameo.relations.apply(lambda x: len(x))
cameo["post"] = cameo.relations.apply(lambda x: "Exhibit Military Posture" in x) #since only few labels, all go to test
post = cameo[cameo.post == True]
cameo = cameo[cameo.post == False]

for row in cameo.iterrows():
    row[1]["relations"].append(str(row[1]["len"])+".")

data = cameo.copy()

from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

mlb = MultiLabelBinarizer()
accept_MLB = mlb.fit_transform(data["relations"])

cols = [f"rel{i}" for i in range(len(accept_MLB[0]))]
data2 = pd.concat([data.reset_index(), pd.DataFrame(accept_MLB, columns = cols)], axis = 1)

#select indexes for train & val
splits = MultilabelStratifiedShuffleSplit(test_size=round(len(data2.text) * 0.4), train_size= (len(data2.text) - round(len(data2.text) * 0.4)), random_state = 0)
test_idx, val_idx = next(splits.split(data2.text, data2[cols]))

pre_split = data2.iloc[val_idx]

splits = MultilabelStratifiedShuffleSplit(test_size=round(len(pre_split.text) * 0.5), train_size= (len(pre_split.text) - round(len(pre_split.text) * 0.5)), random_state = 0)
#0,4 works decent
val_idx, train_idx = next(splits.split(pre_split.text, pre_split[cols]))

test = data2.iloc[test_idx]
val = pre_split.iloc[val_idx] #pre_split
train = pre_split.iloc[train_idx]

train_cam = train[["text","label"]]
val_cam = val[["text","label"]]
test = test[["text","label"]]
test = pd.concat([test, post[["text","label"]]])

print("train shape", train_cam.shape)
print("val shape", val_cam.shape)
print("test shape", test.shape)

relation = []
for row in full.iterrows():
    rel_iter = row[1]["label"] + " <triplet>"
    all_rels = re.findall("(?<=<obj> ).*?(?= <triplet>)", rel_iter)
    relation.append(all_rels)
full["relations"] = relation
full["len"] = full.relations.apply(lambda x: len(x))

idx = []
for row in full.iterrows():
    if "Engage in unconventional mass violence" in row[1]["relations"]: idx.append(row[0])
#Drop unconventional mass violence because not represented 
full = full.drop(index = idx).reset_index().drop(columns = ["index"])
full = full.reset_index().drop(columns = ["index"])

for row in full.iterrows():
    row[1]["relations"].append(str(row[1]["len"])+".")

data = full.copy()

from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

mlb = MultiLabelBinarizer()
accept_MLB = mlb.fit_transform(data["relations"])

cols = [f"rel{i}" for i in range(len(accept_MLB[0]))]
data2 = pd.concat([data.reset_index(), pd.DataFrame(accept_MLB, columns = cols)], axis = 1)

#select indexes for train & val
splits = MultilabelStratifiedShuffleSplit(test_size=round(len(data2.text) * 0.9), train_size= (len(data2.text) - round(len(data2.text) * 0.9)), random_state = int(seed))
val_idx, train_idx = next(splits.split(data2.text, data2[cols]))

val = data2.iloc[val_idx]
train = data2.iloc[train_idx]

train = train[["text","label"]]
train = pd.concat([train, train_cam])
val = val[["text","label"]]
val = pd.concat([val, val_cam])

print("final shapes:")
print("train shape", train.shape)
print("val shape", val.shape)
print("test shape", test.shape) #from CAMEO

#Create entity dictionary for swaps

val_test = pd.concat([val,test])
print("\ncreating entity dictionary for entity swaps...")
ent_dict = {"PERSON":[],"GPE":[],"NORP":[],"EVENTS":[],"FAC":[],"LAW":[],"ORG":[]}

for row in val_test.iterrows():

    subj = re.findall("(?<=<triplet> ).*?(?= <subj>)", row[1]["label"])
    obj = re.findall("(?<=<subj> ).*?(?= <obj>)", row[1]["label"])

    all_ents = subj + obj

    for sub in all_ents:

        ent_type = [word.ent_type_ for word in nlp(sub)]
        if "PERSON" in ent_type: ent_dict["PERSON"].append(sub)
        elif "GPE" in ent_type: ent_dict["GPE"].append(sub)
        elif "NORP" in ent_type: ent_dict["NORP"].append(sub)
        elif "EVENTS" in ent_type: ent_dict["EVENTS"].append(sub)
        elif "FAC" in ent_type: ent_dict["FAC"].append(sub)
        elif "LAW" in ent_type: ent_dict["LAW"].append(sub)
        elif "ORG" in ent_type: ent_dict["ORG"].append(sub)

    if row[0] % 1000 == 0: print("done with ", row[0])

print("entity dictionary created")

# remove duplicated entities to decrease appeareance of common entities
ent_dict_deduped = {"PERSON":[],"GPE":[],"NORP":[],"EVENTS":[],"FAC":[],"LAW":[],"ORG":[]}
for key in ent_dict.keys():
    print(f"{key} found {len(ent_dict[key])}, including duplicates; {len(list(set(ent_dict[key])))} without duplicates")
    ent_dict_deduped[key] = list(set(ent_dict[key]))

#categories to augment
train["inv"] = train.label.apply(lambda x: "Investigate" in x)
train["yield"] = train.label.apply(lambda x: "Yield" in x)
train["intend"] = train.label.apply(lambda x: "Express Intend to Cooperate" in x)
train["mats"] = train.label.apply(lambda x: "Engage In Material Cooperation" in x)
train["coerce"] = train.label.apply(lambda x: "Coerce" in x)
train["demand"] = train.label.apply(lambda x: "Demand" in x)
train["appeal"] = train.label.apply(lambda x: "Appeal" in x)
train["threat"] = train.label.apply(lambda x: "Threaten" in x)
train["protest"] = train.label.apply(lambda x: "Protest" in x)
train["assault"] = train.label.apply(lambda x: "Assault" in x)
train["aid"] = train.label.apply(lambda x: "Provide Aid" in x)
train["rej"] = train.label.apply(lambda x: "Reject" in x)
train["post"] = train.label.apply(lambda x: "Exhibit Military Posture" in x)

#dataframes for labels to augment
data_inv = train[(train.inv == True)]
data_yield = train[(train["yield"] == True)]
data_intend = train[(train.intend == True)]
data_mats = train[(train.mats == True)]
data_coerce = train[(train.coerce == True)]
data_demand = train[(train.demand == True)]
data_appeal = train[(train.appeal == True)]
data_threat = train[(train.threat == True)]
data_protest = train[(train.protest == True)]
data_assault = train[(train.assault == True)]
data_aid = train[(train.aid == True)]
data_rej = train[(train.rej == True)]
data_post = train[(train.post == True)]

aug_df = pd.DataFrame(columns = ["text","label"])

#create list of dataframes that need augmentation
df_list = [data_aid, data_yield, data_inv, data_assault, data_appeal, data_intend, data_demand, data_threat, data_protest, data_rej, data_post] 


#Reverse verb dictionary to be able to query the indicating the verb in the text
verb_dict_rev = {}
for key, value in verb_dict.items():
    verb_dict_rev.setdefault(value, []).append(key)

#Initialize bert-large for embeddings 
#(ConfliBERT performed worse; possibly because not safed in Huggingface for feature extraction)
pipeline = pipeline('feature-extraction', model='bert-large-uncased', device = 2)

def extract_verb(list):
    verb_list = [sub[1] for sub in list]
    return verb_list

import warnings
warnings.filterwarnings('ignore')
df_aug = pd.DataFrame(columns = ["text","label"])
print("\nswapping entities...")
for idx, sub_df in enumerate(df_list):

    if idx < 5:
        its = 1
    else: its = 2

    for it in range(its):

        new_sent = []
        for row in sub_df.iterrows():

            sent = row[1]["text"]
            label = row[1]["label"]

            #extract subjects and objects from labels
            subj = re.findall("(?<=<triplet> ).*?(?= <subj>)", row[1]["label"])
            obj = re.findall("(?<=<subj> ).*?(?= <obj>)", row[1]["label"])

            #draw new entity to replace with; make sure its not drawn twice
            taken_ents = []
            for sub in subj:
                new_subj = draw_new_ent(sub, taken_ents)
                taken_ents.append(new_subj)
                sent = sent.replace(sub, new_subj)
                label = label.replace(sub, new_subj)
            
            for sub in obj:
                new_obj = draw_new_ent(sub, taken_ents)
                taken_ents.append(new_obj)
                sent = sent.replace(sub, new_obj)
                label = label.replace(sub, new_obj)
            
            new_sent.append([sent, label])
        
        sub = pd.DataFrame(new_sent, columns = ["text","label"])

        print("done swapping entities for sub df")

        #verb synonym swaps
        print("\nstarting to replace verbs in sub df...")
        new_verb = []
        for row in sub.iterrows():
            doc = nlp(row[1]["text"])
            replace_dict = {}

            for possible_verb in doc:
                if possible_verb.pos == VERB:
                    check = np.array(pipeline(possible_verb.lemma_))[0,1,:].reshape(1,-1)
                    best5 = [[0,""],[0,""],[0,""],[0,""],[0,""]]
                    if possible_verb.lemma_ in verb_dict.keys() or possible_verb.text in verb_dict.keys():
                        try: relation = verb_dict[possible_verb.lemma_]
                        except: relation = verb_dict[possible_verb.text]
                        for verb in verb_dict_rev[relation]:
                            lem = lemmatizer.lemmatize(verb,"v")
                            best5_lem = [lemmatizer.lemmatize(verb,"v") for verb in extract_verb(best5)]
                            if lem != possible_verb.lemma_ and lem not in best5_lem:
                                sim = cosine_similarity(check, np.array(pipeline(verb))[0,1,:].reshape(1,-1))
                                if sim > best5[0][0]:
                                    best5[0] = [sim, verb]
                                    best5 = sorted(best5, key= lambda x: x[0])
                        replace_dict[possible_verb.text] = np.random.choice(extract_verb(best5))
                    else: print(possible_verb.text, " not in verb dict")

            text = row[1]["text"]
            for verb in replace_dict.keys():
                text = text.replace(verb, replace_dict[verb]).replace("# ","").replace("-","").replace("*","").replace("+","").replace("(p","")
            new_verb.append([text, row[1]["label"]])

        print("finished swapping verbs in sub df!")
        df_aug = pd.concat([df_aug, pd.DataFrame(new_verb, columns = ["text","label"])])

df_full = pd.concat([train, df_aug])

df_full = df_full.reset_index().drop(columns = ["index"])
df_full[["text","label"]].to_csv(f"data_src/annotated/new_train_aug_{seed}.csv")

val = val.reset_index().drop(columns = ["index"])
val[["text","label"]].to_csv(f"data_src/annotated/new_val_aug_{seed}.csv")

test = test.reset_index().drop(columns = ["index"])
test[["text","label"]].to_csv(f"data_src/annotated/new_test_aug_{seed}.csv")
print("done :)")
# %%
