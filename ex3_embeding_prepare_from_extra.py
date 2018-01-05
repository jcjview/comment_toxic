#encoding=utf-8

import re
import sys
import codecs
import os.path
import glob

import gensim
import pandas as pd
import numpy as np
import h5py
import pickle
import sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
i = 1
# word_dict = {}
# with open('word_dict.pkl3', 'rb') as fpkl:
#     word_dict=pickle.load(fpkl)
from config import  *


train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)

list_sentences_train = train["comment_text"].fillna("_na_").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("_na_").values

tokenizer = Tokenizer(num_words=MAX_FEATURES)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = pad_sequences(list_tokenized_train, maxlen=MAX_TEXT_LENGTH)
X_te = pad_sequences(list_tokenized_test, maxlen=MAX_TEXT_LENGTH)

word_index = tokenizer.word_index
def get_coefs(word,*arr):
    try:
        return word, np.asarray(arr, dtype='float32')
    except:
        return word,np.random.normal(size=(embedding_dims,1))

embeddings_index={}

EMBEDDING_FILE='e:\\github\\toxic\\data\\GoogleNews-vectors-negative300.bin'
from gensim.models.keyedvectors import KeyedVectors
word_vectors = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)  # C binary format
print(word_vectors.vector_size)

for word in word_index.keys():
    if word in word_vectors.wv:
        vec=word_vectors.wv[word]
        embeddings_index[word]=vec
all_embs = np.stack(embeddings_index.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()
nb_words = min(MAX_FEATURES, len(word_index))
embedding_matrix = np.random.normal(loc=emb_mean,scale=emb_std,size=(nb_words, embedding_dims))

for word, i in word_index.items():
    if i >= MAX_FEATURES: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

np.savez("w2v_embedding_layer.npz",embedding_matrix)
print ("save word2vec weights OK")
