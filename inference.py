#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 17:21:57 2021

@author: samiulla
"""


import pandas as pd
import os
import re
import warnings
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from nltk.tokenize import sent_tokenize, word_tokenize
import os
import sys

warnings.filterwarnings("ignore")


stop_words = set(stopwords.words('english'))

w2v = Word2Vec.load("./self_embeddings.npy")


def preprocess(text):
    text = text.lower()
    text = re.sub('\n',' ',text)
    text = re.sub('\@highlight',' ',text)
    text = re.sub('\(cnn\) -- ',' ',text)
    text = re.sub('\s+',' ',text)
    return text


def extract_test(file):
    with open(file,'r') as f:
        passage = f.read()
    passage = preprocess(passage)
    return passage


def generate_summary_sam(file,top_n = 5):
    passage = extract_test(file)
    passage = sent_tokenize(passage)
    passage = [x.split(' ') for x in passage]
    sentence_embeddings=[[w2v[word][0] for word in words if word in w2v.wv.vocab and word not in stop_words] for words in passage]
    max_len=max([len(tokens) for tokens in passage])
    sentence_embeddings=[np.pad(embedding,(0,max_len-len(embedding)),'constant') for embedding in sentence_embeddings]
    sim_mat = np.zeros([len(passage), len(passage)])

    for i in range(len(passage)):
      for j in range(len(passage)):
        if i != j:
          sim_mat[i][j] = cosine_similarity(sentence_embeddings[i].reshape(1,max_len), sentence_embeddings[j].reshape(1,max_len))[0,0]

    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank_numpy(nx_graph)
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(passage)), reverse=True)
    summarize_text = []

    if top_n > len(passage):
        top_n = len(passage)

    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentences[i][1]))
    summary = ". ".join(summarize_text)
    return summary


file = sys.argv[1]
top_n = int(sys.argv[2])

summary = generate_summary_sam(file,top_n)

print("Summary : \n" , summary)














