


import sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers

import config

path = './data/'
comp = ''
# EMBEDDING_FILE=path+'glove.6B.50d.txt'
EMBEDDING_FILE='d:\\Users\\xj\\Downloads\\glove-840b-tokens-300d-vectors\\glove.840B.300d.txt'
TRAIN_DATA_FILE=path+'train.csv'
TEST_DATA_FILE=path+'test.csv'



train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)

list_sentences_train = train["comment_text"].fillna("_na_").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("_na_").values

embed_size = config.embedding_dims # how big is each word vector
max_features = config.MAX_FEATURES # how many unique words to use (i.e num rows in embedding vector)
maxlen = config.MAX_TEXT_LENGTH # max number of words in a comment to use


tokenizer = Tokenizer(filters='"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
# list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = pad_sequences(list_tokenized_train)
# X_te = pad_sequences(list_tokenized_test)
word_index = tokenizer.word_index

print(len(tokenizer.word_counts))
print(tokenizer.document_count)
print(len(tokenizer.word_index))
print(len(tokenizer.word_docs))

# np.savetxt(config.train_token_path, X_t, delimiter=',')
# np.savetxt(config.test_token_path, X_te, delimiter=',')

with open(config.train_token_path,'w',encoding='utf-8') as outf:
    for line in list_tokenized_train:
        for w in line:
            outf.write(w)
            outf.write(" ")
        outf.write("\n")
# with open(config.test_token_path,'w',encoding='utf-8') as outf:
#     for line in list_tokenized_test:
#         outf.write(line)
#         outf.write("\n")