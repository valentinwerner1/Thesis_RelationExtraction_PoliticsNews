# In[1]:
#General libraries
import numpy as np
import pandas as pd
import sys
import os 
import re

#Libraries for parsing and getting text from websites
from codecs import xmlcharrefreplace_errors
import feedparser
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

def full_scrape_cnn():
    """iterates through entire available worlds news RSS feeds from guardian,
    will extract text if not extracted yet"""
    feed = feedparser.parse("http://rss.cnn.com/rss/edition_world.rss")
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
                content = soup.find_all("script")
                for part in content:
                    result = re.search('\"articleBody\":\".*\"author\"', part.text)
                    if result != None:
                        text = re.sub('\"articleBody\":\"', "", result.group(0))
                        text = re.sub('\",\"author\"', "", text)
                        author = re.search('\"author\":.*\"dateCreated\"', part.text).group(0)
                        author = re.sub('\"author\":',"",author)
                        author = re.sub(',\"dateCreated\"',"",author)
                if text == "":
                    print(f"could not scrape article {title}")
                else: articles.append([text, author, title])

        except AttributeError:
            print(f"scraper did not work for {item['title']}")

    return articles

# In[3]:

# Load existing DataFrame of cnn articles
df = pd.read_csv(r"C:\Users\svawe\Thesis_RelationExtraction_PoliticsNews\data_src\cnn\cnn2.csv", index_col = 0)

#Scraping cnn news articles
new_df = pd.DataFrame(full_scrape_cnn(), columns = ["text","author","title"])

df = pd.concat([df, new_df], ignore_index = True)

df.to_csv(r"C:\Users\svawe\Thesis_RelationExtraction_PoliticsNews\data_src\cnn\cnn2.csv")
print(f"{new_df.shape[0]} articles have been added")
# %%