#General libraries
import numpy as np
import pandas as pd
import sys
import os 
import re
from dateutil.parser import parse
import datetime 
import json

#Libraries for parsing and getting text from websites
from codecs import xmlcharrefreplace_errors
import feedparser
import urllib.parse
import requests
from bs4 import BeautifulSoup
import ssl

#Libraries for coref
import spacy
import crosslingual_coreference
from crosslingual_coreference import Predictor
import en_core_web_sm

#Loading extras for parsing
ssl._create_default_https_context = ssl._create_unverified_context #avoiding SSL errors
headers =  {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:100.0) Gecko/20100101 Firefox/100.0"} #avoiding some bot-shields
predictor = Predictor(language="en_core_web_sm", device=-1, model_name="spanbert")

import torch
predictor = Predictor(language="en_core_web_sm", device=-1, model_name="spanbert")

df = pd.read_csv("intermediate_full_articles-pre26-11.csv")

coref = []
for row in df.iterrows():
    try: coref.append(predictor.predict(row[1]["full_text"])["resolved_text"])
    except: coref.append(row[1]["full_text"])
    if row[0] % 500 == 0: print(row[0])

df["text"] = coref

df[["paper","link","text"]].to_csv("articles_url_coref.csv")