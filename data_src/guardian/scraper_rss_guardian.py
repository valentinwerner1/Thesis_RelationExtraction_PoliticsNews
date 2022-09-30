#!/usr/bin/env python
# coding: utf-8

# In[1]:
#General libraries
import numpy as np
import pandas as pd
import sys
import os 
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
def scrape_text(link, name, attrs):#, number):
    """retrieves text from the respective article; input is based on data which is received from rss feed"""
    response = requests.get(link, headers = headers) 
    soup = BeautifulSoup(response.content, "html.parser")
    content = soup.find(name, attrs)#[number]
    return content.text

def full_scrape_guardian():
    """iterates through entire available worlds news RSS feeds from guardian,
    will extract text if not extracted yet"""
    feed = feedparser.parse("https://www.theguardian.com/world/rss")
    entry = feed.entries

    articles = []

    for item in entry:
        try:
            author =  item["authors"]
            title = item["title"]
            if title in df.title.to_list(): 
                print("article received already")
            else:
                txt = scrape_text(item["link"], "div", "dcr-5gcz4n")
                articles.append([txt, author, title])

        except AttributeError:
            print(f"scraper did not work for {item['title']}")

    return articles
    

# In[3]:

# Load existing DataFrame of guardian articles
df = pd.read_csv(r"C:\Users\svawe\Thesis_RelationExtraction_PoliticsNews\data_src\guardian\guardian_29092022.csv", index_col = 0)

#Scraping guardian news articles
new_df = pd.DataFrame(full_scrape_guardian(), columns = ["text","author","title"])

df = pd.concat([df, new_df], ignore_index = True)

df.to_csv(r"C:\Users\svawe\Thesis_RelationExtraction_PoliticsNews\data_src\guardian\guardian_29092022.csv")
print(f"{new_df.shape[0]} articles have been added")

# %%
