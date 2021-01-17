
## Librairies

import pandas as pd
import numpy as np
import os
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
# % matplotlib inline
from collections import Counter
import seaborn as sns
matplotlib.rcParams['figure.figsize'] = (20.0, 10.0)
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook
output_notebook()
stemmer = FrenchStemmer()
from nltk.stem.snowball import FrenchStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import matutils, models
import scipy.sparse
from keras.preprocessing.text import Tokenizer
import folium
import pickle
from folium.plugins import HeatMap
import glob
import matplotlib.dates as mdates
import datetime as dt

nltk.download('wordnet')
# spacy for lemmatization
import spacy

from nltk.corpus import stopwords


def clean_X(X):
    #X column data
    pd_list = list(X)
    pd_list = [it.lower() for it in pd_list] # to lowercase
    pd_list = [re.sub(r'ô','o',it) for it in pd_list] 
    pd_list = [re.sub(r'[éêè]','e',it) for it in pd_list] 
    pd_list = [re.sub(r'([^a-zéèûôàêô])',' ',it) for it in pd_list]  #not part
    pd_list = [re.sub('\n', '',it) for it in pd_list]
    
    return pd_list

def tokenization(X_list):
    
    tokenized_df = [word_tokenize(it) for it in X_list]
   
    return tokenized_df

#We added this list to give more sens to our results.
other_stopwords = ['les','a','afin','alors',  'plus', 'moins', 'faut', 'tout', 'tous', 'prends', 'tre', 'si','non','doit','avoir','comme','trop','leurs','faire','ils','peut','bien','aussi','cela','gens', 'sans', 'car', 'très', 'fait', 'nan','\'', 'qu\'', 'd\'', 'l\'', '"', 'être', 'mai', 'faudrait']

stopWords = stopwords.words('french')
stopWords.extend(other_stopwords)
stopWords = set(stopWords)


def rmStopwords(X_list):
    filtered_pd =[]
    for it in X_list:
        
        temp = []
        for word in it:
            if(word not in stopWords):
                temp.append(word)
        filtered_pd.append(temp)
        temp=[]
     
    return filtered_pd

# lemmetization

def lemmetize(column_data):
    nlp = spacy.load('fr', disable=['parser', 'ner'])
    rep_lemmas = []
    for i, reponse in enumerate(column_data):
        if type(reponse) == str:
            doc = nlp(reponse)
            lemmas = [token.lemma_ for token in doc if token.pos_ in ['NOUN', 'PROPN']]
        else:
            lemmas = ''
        rep_lemmas.append(' '.join(lemmas))
 
    return rep_lemmas


def N_mostFreq(X_filtred , n):
    frequency = Counter()
    for it in X_filtred:
        for word in it:
            frequency[word] +=1
    _x = frequency.most_common(n)
    print(_x)


def word_cloud(X_filtred):
    frequency = Counter()
    
    for word in X_filtred:
        frequency[word] +=1
      
    wordcloud = WordCloud(max_font_size=100, max_words=30,background_color="white", colormap="Dark2")
    wordcloud.generate_from_frequencies(frequencies = frequency)
    plt.figure(figsize=(10,6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


# https://gist.github.com/susanli2016/e5d7e2c8bfd9b63523d931a08468d5c4
def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


def bigrams(X,n):
    # X : df column
    # n : number of ngrams
    common_words = get_top_n_bigram(X, n)
    bigram_df = pd.DataFrame(common_words, columns = ['data' , 'count'])
    bigram_df.head() 
    bigram_df = pd.DataFrame(bigram_df.groupby('data').sum()[['count']].sort_values(by= ['count'],ascending=False))
    bigram_df['data'] = bigram_df.index
    # plot the data
    
    plt.figure(figsize=(16,9))
    index = np.arange(len(bigram_df['data']))
    
    plt.bar(index, bigram_df['count'])
    # sns.barplot(x=index, y=bigram_df['count'])
    plt.ylabel('Frequency', fontsize=6)
    plt.xticks(index, bigram_df['data'], fontsize=12, rotation=90)
    plt.title('Top {} bigrams in review'.format(n))
    plt.show()



def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def trigrams(X,n):
    
    # X : df column
    # n : number of ngrams
    
    common_words = get_top_n_trigram(X, n)
    trigram_df = pd.DataFrame(common_words, columns = ['data' , 'count'])
    trigram_df = pd.DataFrame(trigram_df.groupby('data').sum()[['count']].sort_values(by= ['count'],ascending=False))
    trigram_df['data'] = trigram_df.index

    plt.figure(figsize=(16,9))
    index = np.arange(len(trigram_df['data']))
    plt.barh(index, trigram_df['count'])
    #sns.barplot(x=index, y=trigram_df['count'])
    plt.xlabel('Frequency', fontsize=5)
    plt.yticks(index, trigram_df['data'], fontsize=12, rotation=30)
    plt.title('Top {} trigrams in review'.format(n))



