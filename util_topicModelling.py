import pandas as pd
import numpy as np
import nltk
import pickle


nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import  word_tokenize


# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.phrases import Phrases, Phraser
from gensim.models import LsiModel
from gensim.models import CoherenceModel,HdpModel


# spacy for lemmatization
import spacy



from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


import re
import os



def tokenization(X_list):
    tokenized_list = [word_tokenize(it) for it in X_list]
    return tokenized_list


    
## This function get the data and the parameters for the topic models and returns LDA model
def get_topics(culumn_data_selected, n_topics=10, passes=20, add_bigrams = False):
    
    #Tokenize the data
    tokenized_data = tokenization(list(culumn_data_selected))
    
    ## Capture frequent bigrams and add them
    if(add_bigrams):
  
    # Build the bigram and trigram models
        bigram = Phrases(tokenized_data, min_count=5, threshold=200) # higher threshold fewer phrases.
        trigram = Phrases(bigram[tokenized_data], threshold=200)  

    # Faster way to get a sentence clubbed as a trigram/bigram
    #The goal of this class is to cut down memory consumption of Phrases, by discarding model state not strictly needed for the bigram detection task.
    #Use this instead of Phrases if you do not need to update the bigram statistics with new documents any more
        bigram_mod = Phraser(bigram)
        trigram_mod = Phraser(trigram)

    # Form Bigrams
        tokenized_data = make_bigrams(tokenized_data, bigram_mod)
    
    # Create Dictionary  
    id2word = corpora.Dictionary(tokenized_data)

    # Create Corpus
    texts = tokenized_data

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
  
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=n_topics, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=passes,
                                           alpha='auto',
                                           per_word_topics=True)
  
    pprint(lda_model.print_topics())
  
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))  
    
    # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=tokenized_data, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)
  
    return lda_model, corpus, id2word



# This functions are used to capture bigrams and trigrams in the text offered by gensim
def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts, trigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]



