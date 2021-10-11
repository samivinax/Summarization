#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 19:26:24 2021

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

warnings.filterwarnings("ignore")

# os.chdir('/home/samiulla/Downloads/dataset')


stop_words = set(stopwords.words('english'))


train_dir = './dataset/stories_text_summarization_dataset_train'

len(os.listdir(train_dir))

# sample = "/home/samiulla/Downloads/dataset/dataset/stories_text_summarization_dataset_train/0a06e1c26ef6cbadd8f508ec2ade220f755de6a8.story"


data = pd.DataFrame(columns = ["Passage","Summary"])


def preprocess(text):
    text = text.lower()
    text = re.sub('\n',' ',text)
    text = re.sub('\@highlight',' ',text)
    text = re.sub('\(cnn\) -- ',' ',text)
    text = re.sub('\s+',' ',text)
    return text


def extract(file):
    with open(file,'r') as f:
        abc = f.read()
    span = re.search(r'@highlight',abc)
    ques, answer = abc[:span.start()],abc[span.start():]
    ques, answer = preprocess(ques), preprocess(answer)
    data.loc[len(data)] = [ques,answer]
    return data

# file = '/home/samiulla/Downloads/dataset/dataset/stories_text_summarization_dataset_test/0a0a4c90d59df9e36ffec4ba306b4f20f3ba4acb.story'

files = os.listdir(train_dir)

for file in files:
    file = os.path.join(train_dir,file)
    data = extract(file)


data.shape
data.head()


# data.to_csv("./final_dataset.csv")
# data = pd.read_csv("./final_dataset.csv")


if 'Unnamed: 0' in data.columns:
    data = data.drop(["Unnamed: 0"],axis =1)

# for i in range(5):
#     print("Review :" ,data['Passage'][i])
#     print('----------------')
#     print("Summary :" ,data['Summary'][i])
#     print('**********************************************************')


# text = "it's a diwali"



data.shape

data.columns

data['Passage'] = data['Passage'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

data['Summary'] = data['Summary'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))



corpus = []

for x,y in zip(data['Passage'],data['Summary']):
    corpus.append(x)
    corpus.append(y)



# corpus = corpus[:50000]

corpus = [x.split() for x in corpus]




w2v=Word2Vec(corpus, size=150, min_count=10, iter=100, compute_loss=True)
# w2v.save("./self_embeddings.npy")

# w2v = Word2Vec.load("./self_embeddings.npy")


w2v.get_latest_training_loss()

# len(w2v.wv.vocab)
# w2v.similarity('king', 'man')
# w2v.similarity('disease','death')





# w2v.similarity('king', 'doctor')



# passage ="He is very good boy. He is going to school daily. He will attend all the exams and get good marks in all. And also he is the topper of the class. He handles class also in the absence of teachers"


def generate_summary_sam(passage,top_n = 5):
    print("Passage: \n", passage)
    passage = sent_tokenize(passage)
    passage = [x.split(' ') for x in passage]
    sentence_embeddings=[[w2v[word][0] for word in words if word in w2v.wv.vocab and word not in stop_words] for words in passage]
    max_len=max([len(tokens) for tokens in passage])
    sentence_embeddings=[np.pad(embedding,(0,max_len-len(embedding)),'constant') for embedding in sentence_embeddings]
    len(sentence_embeddings[0])
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

    print('----------------')
    print("Summarize Text: \n", ". ".join(summarize_text))


# Inference Code

def extract_test(file):
    with open(file,'r') as f:
        passage = f.read()
    passage = preprocess(passage)
    return passage

def generate_summary_sam(file,top_n = 5):
    passage = extract_test(file)
    print("Passage: \n", passage)
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
        # Step 5 - Offcourse, output the summarize texr

    print('----------------')
    print("Summarize Text: \n", ". ".join(summarize_text))



# generate_summary_sam(file,top_n = 5)

