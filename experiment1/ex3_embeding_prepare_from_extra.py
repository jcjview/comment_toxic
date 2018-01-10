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


# train = pd.read_csv(TRAIN_DATA_FILE)
# test = pd.read_csv(TEST_DATA_FILE)

# y = train[CLASSES_LIST].values

# list_sentences_train = train["comment_text"].fillna("_na_").values
# list_sentences_test = test["comment_text"].fillna("_na_").values


embeddings_index={}
#word2vec
"""
EMBEDDING_FILE='e:\\github\\toxic\\data\\GoogleNews-vectors-negative300.bin'
from gensim.models.keyedvectors import KeyedVectors
word_vectors = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)  # C binary format
print(word_vectors.vector_size)
all_embeddings=word_vectors.wv

"""

#glove

"""
def get_coefs(word,*arr):
    return word, np.asarray(arr, dtype='float32')

def get_coefs(word,*arr):
    try:
        return word, np.asarray(arr, dtype='float32')
    except:
        print('error ',word)
        return word,np.random.normal(size=(embedding_dims,1))

fp=open(GLOVE_EMBEDDING_FILE,encoding='utf-8')
fp.readline()
glove_embeddings = dict(get_coefs(*o.strip().split()) for o in fp)
fp.close()
print(len(glove_embeddings))
all_embeddings=glove_embeddings

for word in word_index.keys():
    if word in all_embeddings:
        vec=glove_embeddings[word]
        embeddings_index[word]=vec
"""

def get_embedding_matrix(word_index):
    def get_coefs(word, *arr):
        if word in word_index and word not in embeddings_index:
            try:
                vector = np.asarray(arr, dtype='float32')
                if len(arr) == embedding_dims:
                    embeddings_index[word] = vector
                else:
                    print(vector.shape)
            except:
                print('error ', word)
                # return word,np.random.normal(size=(embedding_dims,1))
    fp = open(GLOVE_EMBEDDING_FILE, encoding='utf-8')
    for o in fp:
        get_coefs(*o.strip().split())
    fp.close()
    all_embs = np.stack(embeddings_index.values())
    print(all_embs.shape)
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    nb_words = min(MAX_FEATURES, len(word_index))
    embedding_matrix = np.random.normal(loc=emb_mean, scale=emb_std, size=(nb_words, embedding_dims))

    for word, i in word_index.items():
        if i >= nb_words: continue
        if word in embeddings_index:
            embedding_matrix[i] = embeddings_index[word]
    return embedding_matrix
if __name__ == '__main__':
    list_sentences_train = open(train_token_path,encoding='utf-8').readlines()
    list_sentences_test = open(test_token_path,encoding='utf-8').readlines()
    print(len(list_sentences_train))
    print(len(list_sentences_test))
    tokenizer = Tokenizer(num_words=MAX_FEATURES)
    tokenizer.fit_on_texts(list(list_sentences_train))
    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
    list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
    X_t = pad_sequences(list_tokenized_train, maxlen=MAX_TEXT_LENGTH)
    X_te = pad_sequences(list_tokenized_test, maxlen=MAX_TEXT_LENGTH)
    embedding_matrix=get_embedding_matrix(tokenizer.word_index)
    np.savez("w2v_embedding_layer1.npz",embedding_matrix)
    print ("save word2vec weights OK")

