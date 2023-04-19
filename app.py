import streamlit as st
import snscrape.modules.twitter as sntwitter
import torch
from googletrans import Translator
import re
from bs4 import BeautifulSoup
import unicodedata
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def tweet_collection(topic,n):
    
    if topic[0]!='#':
        topic = '#'+topic
        
    tweets = []
    scraper = sntwitter.TwitterSearchScraper(topic)
    c = 0
    for tweet in scraper.get_items():
        if c>=n:
            break;
        data=[tweet.content]
        tweets.append(data)
        c= c+1
    df = pd.DataFrame(tweets,columns=["content"])
    return df

def pre_process(df):
    
    def translate(x):
        translator = Translator()
        text_to_translate = translator.translate(x,dest= 'en')
        return text_to_translate.text
    
    contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how does",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
" u ": " you ",
" ur ": " your ",
" n ": " and "}
    def expand(x):
        for key in contractions:
            value = contractions[key]
            x = x.replace(key, value)
        return x;
    
    def removepunc(x):
        punctuation = "!”$%&’()*+-/:;<=>[]^_`{|}~•@#"
        return x.translate(str.maketrans('','',punctuation))
    
    def removen(x):
        return x.replace("\n","")
    def removes(x):
        return x.replace("&","and")
    
   
    def remove_accented_chars(x):
        x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return x
    
    def remove_repeated_chars(x):
      return re.sub(r'(.)\1{2,}', r'\1',x)

    df['content'] = df['content'].apply(lambda x: x.lower())
    df['content'] = df['content'].apply(lambda x: translate(x))
    df['content'] = df['content'].apply(lambda x: expand(x))
    df['content'] = df['content'].apply(lambda x: re.sub(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '', x))
    df['content'] = df['content'].apply(lambda x: re.sub('RT', "", x))
    df['content'] = df['content'].apply(lambda x: BeautifulSoup(x, 'lxml').get_text())
    df['content'] = df['content'].apply(lambda x: removepunc(x))
    df['content'] = df['content'].apply(lambda x: removen(x) )
    df['content'] = df['content'].apply(lambda x: removes(x) )
    df['content'] = df['content'].apply(lambda x: remove_accented_chars(x))
    df['content'] = df['content'].apply(lambda x: remove_repeated_chars(x))
    
    s = ''
    for i in df['content']:
        s+=i
    return s

@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("rittik9/Pegasus-finetuned-tweet-summary")
    model = AutoModelForSeq2SeqLM.from_pretrained("rittik9/Pegasus-finetuned-tweet-summary").to(device)
    return tokenizer, model


st.set_page_config(layout="wide")
choice = st.sidebar.selectbox("Select your choice", ["Summarize Text", "Summarize Twitter Trending Topic"])

if choice == "Summarize Text":
    st.subheader("Summarize")
    input_text = st.text_area("Enter your text here")
    
    if input_text is not None:
        if st.button("Summarize Text"):
            col1, col2 = st.columns([1,1])
            with col1:
                st.markdown("**Your Input Text**")
                st.info(input_text)
            with col2:
                st.markdown("**Summary Result**")
            
                tokenizer,model=load_model_and_tokenizer()
                inputs = tokenizer(input_text, max_length=1024,return_tensors="pt").to(device)
                # Generate Summary
                summary_ids = model.generate(inputs["input_ids"])
                #summary_ids = model.generate(inputs["input_ids"], max_length=num_tokens, early_stopping=True)
                result=tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                st.success(result)
                
if choice == "Summarize Twitter Trending Topic":
    st.subheader("Summarize Twitter Trending Topic")
    input_text = st.text_area("Enter a hashtag here")
    n = st.text_area("Enter the no. of tweets(1 to 50) you want to summarize")
    
    if input_text is not None:
        if st.button("Summarize"):
                st.markdown("**Summary Result**")
                n = int(n)
                df = tweet_collection(input_text,n)
                s=pre_process(df)
            
                tokenizer,model=load_model_and_tokenizer()
                inputs = tokenizer(s, max_length=1024,return_tensors="pt").to(device)
                # Generate Summary
                summary_ids = model.generate(inputs["input_ids"],max_length=300)
                result=tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                st.success(result)

st.markdown("<h1 style='text-align: center;'>Summarization App By Rittik Panda</h1>", unsafe_allow_html=True)