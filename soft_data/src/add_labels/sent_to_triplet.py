import spacy
import xml.etree.ElementTree as ET
from spacy.symbols import nsubj, dobj, pobj, iobj, neg, xcomp, VERB
from gensim.parsing.preprocessing import strip_multiple_whitespaces
import pandas as pd
import re
import os
import sys

def main():
    nlp = spacy.load('en_core_web_lg')
    read = read_lines(sys.argv[1])

    verb_dict, spec_dict, spec_code = verb_code_dict(sys.argv[2], sys.argv[3])

    found = []
    for line in read:
        if get_triples(line, verb_dict, spec_dict, spec_code, nlp) != " ".join([]):
            found.append([line, get_triples(line, verb_dict, spec_dict, spec_code, nlp)])
    
    df = pd.DataFrame(found, columns = ["text", "label"])


    # df = pd.DataFrame([[line, " ".join(get_triples(line, verb_dict, spec_dict, spec_code, nlp))] for line in read if get_triples(line, verb_dict, spec_dict, spec_code, nlp) != []],
    #                  columns = ["text", "label"])

    df.to_csv(sys.argv[4])

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
                        verb_dict[line_s[0].lower()] = main_codes[code] 
                        for word in line_s[1].split():
                            verb_dict[word.lower()] = main_codes[code]
                    else:
                        word = re.split("\[|\]", line)[0]
                        verb_dict[word.lower()] = main_codes[code]
            else:
                if cur_main_code != "--":
                    if "{" in line:         #e.g. "HURRY {HURRIES HURRYING HURRIED }" 
                        line_s = re.split("\{|\}", line)    #split at { and }
                        verb_dict[line_s[0].lower()] = main_codes[cur_main_code]
                        for word in line_s[1].split():
                            verb_dict[word.lower()] = main_codes[cur_main_code]
                    else:                   #only single words with sometimes comments, e.g.: CENSURE  # JON 5/17/95
                        word = line.split("#")[0].rstrip()    #gets part before "#", removes all whitespaces to the right
                        verb_dict[word.lower()] = main_codes[cur_main_code]

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
                    spec_dict[verb_match] = poss
            except:
                count += 1

    print(f"{count} patterns could not be loaded")        

    return verb_dict, spec_dict, spec_code


def get_triples(sentence, verb_dict, spec_dict, spec_code, nlp):
    """create triplet structure for training from text input, 
    verb_dict needs to be loaded before,
    spacy model needs to be initialized before """
    doc = nlp(sentence)
    verbs = []
    dict = {}


    for possible_verb in doc:
        if possible_verb.pos == VERB:
            if neg in [child.dep for child in possible_verb.children]: continue
            else: 
                for possible_subject in possible_verb.children: 
                    if possible_subject.dep == xcomp:   #subj / obj of composed verb should also be subj / obj of main verb
                        main_verb = possible_subject
                        main_idx = possible_subject.idx
                        for token in doc.ents:
                            if token.label_ in ["GPE", "NORP", "EVENTS", "FAC", "LAW", "ORG", "PERSON"]:
                                if token.root.dep_ == "poss":
                                    if token.root.head.head.idx == possible_verb.idx:
                                        verbs.append([main_idx, main_verb.lemma_, token.text, token.root.head.dep_])
                                else:
                                    if token.root.head.idx == possible_verb.idx:
                                        verbs.append([possible_verb.idx, possible_verb.lemma_, token.text, token.root.dep_])

                for token in doc.ents:
                    if token.label_ in ["GPE", "NORP", "EVENTS", "FAC", "LAW", "ORG", "PERSON"]:
                        if token.root.dep_ == "poss":
                            if token.root.head.head.idx == possible_verb.idx:
                                verbs.append([possible_verb.idx, possible_verb.lemma_, token.text, token.root.head.dep_])
                        else:
                            if token.root.head.idx == possible_verb.idx:
                                verbs.append([possible_verb.idx, possible_verb.lemma_, token.text, token.root.dep_])



    # doc = nlp(sentence)
    # verbs = []
    # dict = {}

    # for possible_verb in doc:           #parses through all words in sentence
    #     if possible_verb.pos == VERB:   #we only care about verbs
    #         if neg in [child.dep for child in possible_verb.children]: continue #we exclude all negated verbs
    #         else: 
    #             for candidate in possible_verb.children: #for composed verbs of verb (e.g. "want to join" -> "want join")
    #                 if candidate.dep == xcomp:   #subj / obj of composed verb should also be subj / obj of main verb
    #                     main_verb = candidate    
    #                     main_idx = candidate.idx
    #                     for chunk in doc.noun_chunks:   #chunks are noun-groups (e.g.: "78 out of 100 people" instead of "people")
    #                         if chunk.root.head.idx == possible_verb.idx:    #if chunk applies to xcomp (want),
    #                                                                         #treat it like it aplles to main verb ("join")
    #                             verbs.append([main_idx, main_verb.lemma_, chunk.text, chunk.root.dep_])
    #                             if main_idx in dict.keys(): dict[main_idx] += 1 #count how often verb is used
    #                             else: dict[main_idx] = 1

    #             for chunk in doc.noun_chunks:       #for normal verbs, check chunks directly
    #                 if chunk.root.head.idx == possible_verb.idx:
    #                     verbs.append([possible_verb.idx, possible_verb.lemma_, chunk.text, chunk.root.dep_])
    #                     if possible_verb.idx in dict.keys(): dict[possible_verb.idx] += 1
    #                     else: dict[possible_verb.idx] = 1
    
    # trip_idx = [key for key in dict if dict[key] > 1]   #if verbs used more than once, its candidate for triplet

    #priority for subj-relation-obj triplets
    mapper = {"nsubj":1,"dobj":2, "pobj":2, "iobj":2}

    #create df from verbs extracted 
    df = pd.DataFrame(verbs, columns = ["idx", "verb", "noun", "noun_type"])
    df["noun_map"] = df.noun_type.map(mapper)  #turn noun_types into priority 

    if df.shape[0] >= 2:
        #create groups that resolve around same word
        gb = df.groupby('idx')   

        matches = []
        for x in gb.groups:
            group = gb.get_group(x).sort_values("noun_map")
            if group.shape[0] == 2:
                if group.noun_type.iloc[0] != group.noun_type.iloc[1]:
                    matches.append([group.iloc[0].noun, group.iloc[0].verb, group.iloc[1].noun])
            elif group.shape[0] > 2:
                for i in range(group.shape[0] - 1):
                    if group.noun_type.iloc[i] != group.noun_type.iloc[i+1]:
                        matches.append([group.iloc[0].noun, group.iloc[0].verb, group.iloc[1].noun])
        
        
        # #only keep groups if verb idx was identified as potential triplet before, sort by priority for structure
        # df_l = [gb.get_group(x).sort_values("noun_map") for x in gb.groups if gb.get_group(x).idx.iloc[0] in dict]
        # matches = [merge_trip(group) for group in df_l if not merge_trip(group) == None] #get groups into triplet structure
        

        #turn matches into triples by only keeping those with coded verbs, return code instead of verb
        triples = []
        for match in matches:
            if match[1].lower() in spec_dict:
                for poss_pattern in spec_dict[match[1].lower()]:
                    if set(poss_pattern.split()).intersection(sentence.split()) == set(poss_pattern.split()):
                        triples.append(f"<triplet> {match[0]} <subj> {match[2]} <obj> {spec_code[poss_pattern]}")
                        
            elif match[1].lower() in verb_dict:
                triples.append(f"<triplet> {match[0]} <subj> {match[2]} <obj> {verb_dict[match[1].lower()]}")
            else: print(f"couldn't match {match[1].lower()}")

        #triples = [f"<triplet> {match[0]} <subj> {match[2]} <obj> {verb_dict[match[1].lower()]}" for match in matches if match[1].lower() in verb_dict]

        full_triple = " ".join(triples)

        return full_triple

# def merge_trip(df):
#     """helper function to turn two rows of a pandas groupby into subj, verb, obj"""
#     if df.shape[0] == 2:
#         if df.noun_type.iloc[0] != df.noun_type.iloc[1]:
#             return [df.iloc[0].noun, df.iloc[0].verb, df.iloc[1].noun]
#     elif df.shape[0] > 2:
#         for i in range(df.shape[0] - 1):
#             if df.noun_type.iloc[i] != df.noun_type.iloc[i+1]:
#                 return [df.iloc[0].noun, df.iloc[0].verb, df.iloc[1].noun]


if __name__ == "__main__":
    main()
    