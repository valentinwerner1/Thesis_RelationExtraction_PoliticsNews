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
#import feedparser
import urllib.parse
import requests
from bs4 import BeautifulSoup
import ssl

#Libraries for coref
import spacy
from gensim import parsing
# import crosslingual_coreference
# from crosslingual_coreference import Predictor
# import en_core_web_sm

#Loading extras for parsing
ssl._create_default_https_context = ssl._create_unverified_context #avoiding SSL errors
headers =  {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:100.0) Gecko/20100101 Firefox/100.0"} #avoiding some bot-shields
#predictor = Predictor(language="en_core_web_sm", device=-1, model_name="spanbert")


df = pd.read_csv("gdelt_combined.csv", index_col = 0)

new_list = []
for row in df.iterrows():
    if "www.bbc.co.uk/news/world" in row[1]["url"]:
        new_list.append([row[1]["paper"], row[1]["url"], row[1]["date"]])
    elif "tass.com/world" in row[1]["url"]:
        new_list.append([row[1]["paper"], row[1]["url"], row[1]["date"]])
    elif "cbc.ca/news/world" in row[1]["url"]:
        new_list.append([row[1]["paper"], row[1]["url"], row[1]["date"]])
    elif "https://ewn.co.za/" in row[1]["url"]:
        new_list.append([row[1]["paper"], row[1]["url"], row[1]["date"]])
    elif "smh.com.au/world" in row[1]["url"]:
        new_list.append([row[1]["paper"], row[1]["url"], row[1]["date"]])

df = pd.DataFrame(new_list, columns = ["paper","url","date"])

print(df.shape)

df_scraped = pd.read_csv("gdelt_reduced_scraped.csv", index_col = 0)
#df_scraped = pd.DataFrame(columns = ["paper","url","text"])
for i in range(165):
    res = []
    for row in df[(df.paper != "nyt") & (df.paper != "jt")].iloc[i*500:(i+1)*500].iterrows():
        #skip nyt and jt because not scrapable without java
        response = requests.get(row[1]["url"], headers = headers) 
        soup = BeautifulSoup(response.content, "html.parser")

        if row[1]["paper"] == "cbc":
            try:
                res.append([row[1]["paper"],row[1]["url"],parsing.strip_multiple_whitespaces(str(soup.find("div", "story").text))])
            except: 
                print(f"couldn't scrape {row[1]['url']} from {row[1]['paper']}")
        elif row[1]["paper"] == "smh":
            try:
                res.append([row[1]["paper"],row[1]["url"],parsing.strip_multiple_whitespaces(str(soup.find("div", "_1665V _2q-Vk").text.replace("\n","")))])
            except: 
                print(f"couldn't scrape {row[1]['url']} from {row[1]['paper']}")
        elif row[1]["paper"] == "bbc":
            try:
                content = soup.find_all("div", "ssrcss-11r1m41-RichTextComponentWrapper ep2nwvo0")
                text = ""
                for para in content:
                    text = text + " " + para.text
                res.append([row[1]["paper"],row[1]["url"],parsing.strip_multiple_whitespaces(str(text.replace("\n","")))])
            except: 
                print(f"couldn't scrape {row[1]['url']} from {row[1]['paper']}")
        elif row[1]["paper"] == "tass":
            try:
                res.append([row[1]["paper"],row[1]["url"],parsing.strip_multiple_whitespaces(str(soup.find("div", "text-block").text.replace("\n","")))])
            except: 
                print(f"couldn't scrape {row[1]['url']} from {row[1]['paper']}")
        elif row[1]["paper"] == "ewn":
            try:
                res.append([row[1]["paper"],row[1]["url"],parsing.strip_multiple_whitespaces(str(". ".join(soup.find("div", "medium-12 columns").text.replace("\n","").split("."))))])
            except: 
                print(f"couldn't scrape {row[1]['url']} from {row[1]['paper']}")
        
    df_n = pd.DataFrame(res, columns = ["paper","url","text"])
    df_scraped = pd.concat([df_scraped, df_n])
    df_scraped.to_csv("gdelt_reduced_scraped.csv")
    print("done with ", i)
        
    df_n = pd.DataFrame(res, columns = ["paper","url","text"])
    df_scraped = pd.concat([df_scraped, df_n])
    df_scraped.to_csv("gdelt_reduced_scraped.csv")
    print("done with ", i)
        