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
TRAIN_DATA_FILE=path+'train.csv'
TEST_DATA_FILE=path+'test.csv'

embed_size = config.embedding_dims # how big is each word vector
max_features = config.MAX_FEATURES # how many unique words to use (i.e num rows in embedding vector)
maxlen = config.MAX_TEXT_LENGTH # max number of words in a comment to use

train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)

list_sentences_train = train["comment_text"].fillna("_na_").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("_na_").values

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

word_index = tokenizer.word_index
def get_coefs(word,*arr):
    try:
        return word, np.asarray(arr, dtype='float32')
    except:
        return word,np.random.normal(size=(embed_size,1))


embedding_matrix = np.load("w2v_embedding_layer.npz")["arr_0"]
file_path = "weights_LSTM_glove.best.hdf5"
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
checkpoint = ModelCheckpoint(file_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor="loss", mode="min", patience=5)
callbacks_list = [checkpoint, early]
model.fit(X_t, y, batch_size=32, epochs=2, callbacks=callbacks_list) # validation_split=0.1);

y_test = model.predict([X_te], batch_size=1024, verbose=1)
sample_submission = pd.read_csv(path+'sample_submission.csv')
sample_submission[list_classes] = y_test
sample_submission.to_csv('submission.csv', index=False)