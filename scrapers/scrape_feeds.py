# In[1]:
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

#Loading extras for parsing
ssl._create_default_https_context = ssl._create_unverified_context #avoiding SSL errors
headers =  {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:100.0) Gecko/20100101 Firefox/100.0"} #avoiding some bot-shields

#selection of newspaper: 3-4 per continent, 1 per country (countries chosen based on GDP on continent & political influence (in case of russia)); 
                        ## factors: - how many readers / circulation
                        ##          - political orientation (as central as possible)
                        ##          - must publish in english 

rss_dict = {
    "bbc": "http://feeds.bbci.co.uk/news/world/rss.xml#",                           # UK
    #"guardian": "https://www.theguardian.com/world/rss",                           # UK
    "spiegel": "https://www.spiegel.de/international/index.rss",                    # GER
    "f24" : "https://www.france24.com/en/rss",                                      # FR
    "tass": "http://tass.com/rss/v2.xml",                                           # RU

    #"post": "https://feeds.washingtonpost.com/rss/world",                          # US
    "nyt": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",                # US
    "cbc": "https://www.cbc.ca/cmlink/rss-world",                                   # CA
    #"ctv": "http://www.ctvnews.ca/rss/World",                                      # CA

    "folha" : "https://feeds.folha.uol.com.br/internacional/en/world/rss091.xml",   # BR
    "bat" : "https://www.batimes.com.ar/feed",                                      # AR

    #"cd" : "http://www.chinadaily.com.cn/rss/world_rss.xml",                       # CN
    "jt" : "https://www.japantimes.co.jp/news_category/world/feed/",                # JP
    "it" : "https://timesofindia.indiatimes.com/rssfeeds/296589292.cms",            # IN

    "independent" : "https://www.egyptindependent.com/feed/",                       # EG
    "ewn" : "http://ewn.co.za/RSS%20Feeds/Latest%20News?category=World",            # ZA

    "smh" : "https://www.smh.com.au/rss/world.xml",                                 # AU
}

os.chdir("..") #puts directory at main


# In[2]:
#Functions for scraping

def get_data(entry, rss, df):
    """get relevant data from rss feeds"""
    data = []   #collects new articles
    dups = 0    #counts duplicates that were not collected again
    for article in entry:
        try: #if any main part is missing, the article is skipped instead of the pipeline breaking
            text = article["title"] 
            if text in df.title.to_list():
                dups += 1
            else: 
                desc = article["summary"]
                desc = BeautifulSoup(desc).get_text()   #strips html tags which some papers are using in the description
                date = article["published"]
                try:
                    authors = ""
                    if rss == "cd": #china daily uses authorname instead of authors
                        for name in article["authorname"]:
                            if authors == "": authors += name["name"]
                            else: authors = authors + "; " + name["name"]
                    else:
                        for name in article["authors"]:
                            if authors == "": authors += name["name"]
                            else: authors = authors + "; " + name["name"]
                except:
                    authors = "not mentioned in feed" #many feeds don't include authors in the feed
                data.append([rss, text, desc, authors, date])
        except:
            print(f"couldn't scrape {article['title']} in {rss}")
    print(f"found {dups} duplicates")
    return data

def parse_feeds(rss_dict, df):
    """parses all rss feeds and cals get_data to retrieve information"""
    articles = []
    for rss in rss_dict:
        feed = feedparser.parse(rss_dict[f"{rss}"])
        entry = feed.entries
        if rss == "tass": #because tass has no dedicated world news feed, we extract only entries that link to world articles
            new_entry = [article for article in entry if "tass.com/world/" in article["link"]]
            data = get_data(new_entry, rss, df)
            for article in data:
                articles.append(article)
            print(f"done with {rss}")
        else:
            data = get_data(entry, rss, df)
            for article in data:
                articles.append(article)
            print(f"done with {rss}")
    print(f"found {len(articles)} new articles")
    return articles
    
def post_process(articles, df):
    """create and transform dataframe: generate main text, article id and process time, drop duplicates and unneeded columns"""
    new_df = pd.DataFrame(articles, columns = ["paper", "title", "summ","authors", "DateTime"])

    #concat title and summary 
    new_df["full_text"] = new_df["title"] + ". " + new_df["summ"]

    #time post processing

    #nyt == GMT, bbc == GMT, spiegel == GMT +2, f24 == GMT, tass == GMT +3, 
    #cbc == GMT -4, folha == GMT -3, bat == GMT, jt == GMT +9, it == GMT +5.5, 
    #independent == GMT, ewn == GMT, smh == GMT + 11
    new_df["date"] = new_df.DateTime.apply(lambda x: parse(x).date())

    tzinfos = {"EDT": -14400, "EST": -14400} #because cbc uses timezone instead of offset
    new_df["date_time"] = new_df.DateTime.apply(lambda x: (parse(x) - parse(x, tzinfos = tzinfos).utcoffset()).replace(tzinfo = None))

    #drop old DateTime row that has been resolved
    new_df = new_df.drop(columns = "DateTime")

    #remove duplicated summaries and titles
    new_df = new_df[~new_df.duplicated(["full_text"])]

    #generate article_id
    if df.shape[0] == 0: new_df["article_id"] = new_df.index
    else: new_df["article_id"] = max(df.article_id) + 1 + new_df.index

    #fill possible NaNs with empty string to avoid errors in jsonl file
    if new_df.full_text.isna().any():
        new_df["full_text"].fillna("", inplace = True)
    
    return new_df

def append_jsonl(new_df):
    """append new dataframe to jsonl file which is input for labeling"""
    
    jsonl = []  #list to collect rows in jsonl format
    for index in new_df.index:
        jsonl.append({"text":new_df.full_text.iloc[index], "article_id":int(new_df.article_id.iloc[index])})

    #turn to jsonl file
    with open("data_src/raw/in_label/all_articles.jsonl", 'a') as f: 
        for item in jsonl:
            f.write(json.dumps(item) + "\n")


# In[3]:
#Execute all functions

# Load existing DataFrame of all articles
df = pd.read_csv("data_src/raw/all_articles.csv", index_col = 0)

#Getting data from rss feeds
new_df = post_process(parse_feeds(rss_dict, df), df)

#Append new data to existing jsonl file
append_jsonl(new_df)

#Concat old and new data and safe as csv of all articles
df = pd.concat([df, new_df], ignore_index = True)
df.to_csv("data_src/raw/all_articles.csv")

print(f"{new_df.shape[0]} articles have been added")

# %%
