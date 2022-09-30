#!/usr/bin/env python
# coding: utf-8

# In[1]:

#General libraries
import numpy as np
import pandas as pd
import sys
import os 
import json
os.environ['PYTHONHASHSEED'] = "0" #making sure it hashes everytime the same thing

#Libraries for parsing and getting text from websites
from codecs import xmlcharrefreplace_errors
import feedparser
import hashlib
import urllib.parse
import requests
from bs4 import BeautifulSoup
import ssl

#Loading extras for parsing
ssl._create_default_https_context = ssl._create_unverified_context #avoiding SSL errors
headers =  {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:100.0) Gecko/20100101 Firefox/100.0"} #avoiding some bot-shields




# In[2]:

#Functions for scraping 
def full_scrape_bbc():
    """iterates through entire available worlds news RSS feeds from bbc,
    will extract text if not extracted yet"""
    feed = feedparser.parse("http://feeds.bbci.co.uk/news/world/rss.xml")
    entry = feed.entries

    articles = []

    for item in entry:
        try:
            title = item["title"]
            if title in df.title.to_list(): 
                print("article received already")
            else:
                response = requests.get(item["link"], headers = headers) 
                soup = BeautifulSoup(response.content, "html.parser")
                content = soup.find_all("div", "ssrcss-11r1m41-RichTextComponentWrapper ep2nwvo0")#[number]
                text = ""
                for para in content:
                    text += para.text
                try:
                    content = soup.find("script")
                    author = json.loads(content.text)["author"]["name"]
                except:
                    author = "BBC News - Couldn't scrape author"

                if text == "":
                    print(f"could not scrape article {title}")
                else: articles.append([text, author, title])
                
        except AttributeError:
            print(f"scraper did not work for {item['title']}")

    return articles

# In[3]:

# Load existing DataFrame of bbc articles
df = pd.read_csv(r"C:\Users\svawe\Thesis_RelationExtraction_PoliticsNews\data_src\bbc\bbc.csv", index_col = 0)

#Scraping bbc news articles
new_df = pd.DataFrame(full_scrape_bbc(), columns = ["text","author","title"])

df = pd.concat([df, new_df], ignore_index = True)

df.to_csv(r"C:\Users\svawe\Thesis_RelationExtraction_PoliticsNews\data_src\bbc\bbc.csv")
print(f"{new_df.shape[0]} articles have been added")
# %%
