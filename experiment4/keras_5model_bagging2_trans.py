'''
The code is tested on Keras 2.0.0 using Theano backend, and Python 3.5
referrence Code:https://github.com/keras-team/keras/blob/master/examples/imdb_cnn_lstm.py
'''

import os
import re

import h5py
import numpy as np
import pandas as pd
# from keras import initializations
import sys
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Input, Embedding, Dropout, RepeatVector, Conv1D, GlobalAveragePooling1D, LSTM, \
    BatchNormalization, merge, PReLU, Bidirectional, GlobalMaxPool1D, Masking, GRU
from keras.layers import recurrent
from keras.models import Model
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.cross_validation import KFold
from sklearn.metrics import log_loss

import config
from  keras_Attention import get_model as get_model_Attention

path = '../data/'
EMBEDDING_FILE = config.embedding_path
TRAIN_DATA_FILE = config.TRAIN_DATA_FILE
TEST_DATA_FILE = config.TEST_DATA_FILE

MAX_SEQUENCE_LENGTH = 150
MAX_NB_WORDS = 100000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1
SPLIT = config.SPLIT
num_lstm = 300
num_dense = 256
rate_drop_lstm = 0.25
rate_drop_dense = 0.25

act = 'relu'

# Regex to remove all Non-Alpha Numeric and space
special_character_removal = re.compile(r'[^a-z\d ]', re.IGNORECASE)
# regex to replace all numerics
replace_numbers = re.compile(r'\d+', re.IGNORECASE)

########################################
## index word vectors
########################################

kernel_name = "5_model_bagging2.0"
perm_path = kernel_name + "perm.npz"
embedding_matrix_path = kernel_name + "_imdb_blstm_embedding_layer.npz"

########################################
## process texts in datasets
########################################
TRAIN_HDF5 = './train_hdf5.h5'


def dump_X_Y_train_test(train, test, y):
    outh5file = h5py.File(TRAIN_HDF5, 'w')
    test_token = outh5file.create_dataset('test_token', data=test)
    train_token = outh5file.create_dataset('train_token', data=train)
    train_label = outh5file.create_dataset(name='train_label', data=y)
    outh5file.close()


def load_tran_test_y():
    outh5file = h5py.File(TRAIN_HDF5, 'r')
    train = outh5file['train_token']
    test = outh5file['test_token']
    y = outh5file['train_label']
    return train, test, y


def get_X_train_X_test(train_df, test_df):
    print('Processing text dataset')
    list_sentences_train = train_df["comment_text"].fillna("NA").values
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y = train_df[list_classes].values
    list_sentences_test = test_df["comment_text"].fillna("NA").values
    comments = []
    for text in list_sentences_train:
        comments.append(text_to_wordlist(text))
    test_comments = []
    for text in list_sentences_test:
        test_comments.append(text_to_wordlist(text))

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(comments + test_comments)

    sequences = tokenizer.texts_to_sequences(comments)
    test_sequences = tokenizer.texts_to_sequences(test_comments)

    word_index = tokenizer.word_index
    print('Found %s unique tokens' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', y.shape)

    test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of test_data tensor:', test_data.shape)
    return data, test_data, tokenizer.word_index


def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.

    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]

    text = " ".join(text)

    # Remove Special Characters
    text = special_character_removal.sub('', text)

    # Replace Numbers
    text = replace_numbers.sub('n', text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return (text)


"""
prepare embeddings
"""


def get_embedding_matrix(word_index):
    #    Glove Vectors
    print('Indexing word vectors')
    embeddings_index = {}
    f = open(EMBEDDING_FILE)
    for line in f:
        values = line.split()
        word = ' '.join(values[:-300])
        coefs = np.asarray(values[-300:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Total %s word vectors.' % len(embeddings_index))
    print('Preparing embedding matrix')
    nb_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    np.save(embedding_matrix_path, embedding_matrix)
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    return embedding_matrix


"""
 define the model structure

 """
# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# LSTM
lstm_output_size = 70


def get_model1(embedding_matrix):#yin
    input1 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    # x = Dropout(rate_drop_dense)(input1)
    embedding_layer = Embedding(MAX_NB_WORDS,
                                EMBEDDING_DIM,
                                input_length=MAX_SEQUENCE_LENGTH,
                                weights=[embedding_matrix],
                                trainable=True)
    input1 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

    x = embedding_layer(input1)
    xconv = Conv1D(filters=128,
                   kernel_size=3,
                   padding='same',
                   activation='relu')(x)
    xconv = GlobalAveragePooling1D()(xconv)
    xlstm = LSTM(350, dropout_W=0.2, dropout_U=0.2)(x)
    xlstm = BatchNormalization()(xlstm)

    x = merge([xconv, xlstm], mode='concat')

    x = Dense(500)(x)
    x = PReLU()(x)
    x1 = BatchNormalization()(x)

    x = Dense(500)(x1)
    x = PReLU()(x)
    x2 = BatchNormalization()(x)
    x = merge([x1, x2], mode='sum')

    x = Dense(500)(x)
    x = PReLU()(x)
    x3 = BatchNormalization()(x)
    x = merge([x1, x2, x3], mode='sum')

    x = Dense(500)(x)
    x = PReLU()(x)
    x4 = BatchNormalization()(x)
    x = merge([x1, x2, x3, x4], mode='sum')

    x = Dense(500)(x)
    x = PReLU()(x)
    x5 = BatchNormalization()(x)
    x = merge([x1, x2, x3, x4, x5], mode='sum')
    # x = concatenate([x1, x2, x3, x4], axis=1)
    x = Dropout(0.2)(x)
    out = Dense(6, activation='sigmoid')(x)
    model = Model(inputs=input1, outputs=out)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def get_model2(embedding_matrix):
    model=get_model_Attention(embedding_matrix)
    return model

def get_model3(embedding_matrix):
    input1 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    # x = Dropout(rate_drop_dense)(input1)
    embedding_layer = Embedding(MAX_NB_WORDS,
                                EMBEDDING_DIM,
                                input_length=MAX_SEQUENCE_LENGTH,
                                weights=[embedding_matrix],
                                trainable=True)
    x=embedding_layer(input1)
    x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=input1, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def get_model4(embedding_matrix):
    input1 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedding_layer = Embedding(MAX_NB_WORDS,
                                EMBEDDING_DIM,
                                input_length=MAX_SEQUENCE_LENGTH,
                                weights=[embedding_matrix],
                                trainable=True)(input1)
    x = Bidirectional(GRU(64, return_sequences=True))(embedding_layer)
    x = Dropout(0.3)(x)
    x = Bidirectional(GRU(64, return_sequences=False))(x)
    x = Dense(32, activation="relu")(x)
    output_layer = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=input1, outputs=output_layer)
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(clipvalue=1, clipnorm=1),
                  metrics=['accuracy'])
    return model

def get_model0(embedding_matrix):
    input1 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    # x = Dropout(rate_drop_dense)(input1)
    embedding_layer = Embedding(MAX_NB_WORDS,
                                EMBEDDING_DIM,
                                input_length=MAX_SEQUENCE_LENGTH,
                                weights=[embedding_matrix],
                                trainable=True)
    x = embedding_layer(input1)
    x = Dropout(rate_drop_dense)(x)
    x = recurrent.LSTM(lstm_output_size)(x)
    x = RepeatVector(MAX_SEQUENCE_LENGTH)(x)
    x = recurrent.LSTM(lstm_output_size)(x)
    x = Dropout(rate_drop_dense)(x)
    out = Dense(6, activation='sigmoid')(x)
    model = Model(inputs=input1, outputs=out)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['mse'])
    # model.summary()
    return model

def get_model(embedding_matrix,func):
    print(func)
    get_afunc = getattr(sys.modules[__name__], func)
    return get_afunc(embedding_matrix)

def submit(y_test, bst_val_score, STAMP):
    sample_submission = pd.read_csv(config.path + "sample_submission.csv")
    sample_submission[config.CLASSES_LIST] = y_test
    sample_submission.to_csv('%.4f_' % (bst_val_score) + STAMP + '.csv', index=False)


def get_Y(train):
    return train[config.CLASSES_LIST].values


########################################################################################################################################################
def mlogloss(true_label, predicted, eps=1e-15):
  p = np.clip(predicted, eps, 1 - eps)
  if true_label == 1:
    return -np.log2(p)
  else:
    return -np.log2(1 - p)

def multi_log_loss(y_true, y_pred):  # score function for CV
    # Handle all zeroes
    print('y_true',y_true.shape)
    print('y_pred',y_pred.shape)

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    print('y_true',y_true.shape)
    print('y_pred',y_pred.shape)

    n_rows = len(y_true)
    score_sum = 0
    for i in range(n_rows):
        score_sum +=mlogloss(y_true[i],y_pred[i])
    score = score_sum / n_rows
    print('n_rows',n_rows)

    return score


if __name__ == '__main__':
    train = pd.read_csv('../data/train_valid_test.csv')
    test = pd.read_csv(TEST_DATA_FILE)
    X_train, X_test, word_index = get_X_train_X_test(train, test)
    if os.path.exists(embedding_matrix_path):
        embedding_matrix = np.load(embedding_matrix_path)["arr_0"]
    else:
        embedding_matrix = get_embedding_matrix(word_index)
    # embedding_matrix,nb_words=get_embedding_matrix(word_index)
    y = get_Y(train)
    ## cv-folds
    nfolds = 10
    folds = KFold(len(y), n_folds=nfolds, shuffle=True, random_state=111)

    ## train models
    i = 0
    nbags = 5
    nepochs = 5
    pred_oob = np.zeros(y.shape)
    pred_test = np.zeros(shape=(X_test.shape[0], 6))
    print('pred_oob', pred_oob.shape)
    print('pred_test', pred_test.shape)
    for (inTr, inTe) in folds:
        xtr = X_train[inTr]
        ytr = y[inTr]
        xte = X_train[inTe]
        yte = y[inTe]
        pred = np.zeros(yte.shape)
        print('xtr', xtr.shape)
        print('xte', xte.shape)
        for j in range(nbags):
            STAMP = kernel_name + '_fold_%d_nbags_%d_' % (i,j)
            print(STAMP)
            early_stopping = EarlyStopping(monitor='val_loss', patience=5)
            bst_model_path = STAMP + '.h5'
            model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, verbose=1, save_weights_only=True)
            model = get_model(embedding_matrix,'get_model'+str(j))
            fit = model.fit(xtr, ytr, batch_size=256, epochs=5, verbose=0, callbacks=[early_stopping, model_checkpoint],
                            validation_data=(xte, yte))
            model.load_weights(bst_model_path)
            pred_temp=model.predict(xte,1024,verbose=1)
            pred+=pred_temp
            score = log_loss(yte, pred_temp)/6
            score1 = multi_log_loss(yte, pred_temp)
            print('%d score'%j,score)
            print('%d score1' % j, score1)
            pred_test += model.predict(X_test, 1024, verbose=1)
            # fit = model.fit_generator(generator=batch_generator(xtr, ytr, 128, True),
            #                           nb_epoch=nepochs,
            #                           # samples_per_epoch=xtr.shape[0],
            #                           verbose=1)
            # pred += model.predict_generator(generator=batch_generatorp(xte, 256, False),
            #                                 # val_samples=xte.shape[0],
            #                                 verbose=1)
            #
            # pred_test += model.predict_generator(generator=batch_generatorp(X_test, 1024, False),
            #                                      # val_samples=X_test.shape[0],
            #                                     verbose = 1)
        pred /= nbags
        pred_oob[inTe] = pred
        score = log_loss(yte, pred)/6
        score1 = multi_log_loss(yte, pred)
        i += 1
        print('Fold ', i, '- log_loss:', score)
        print('Fold ', i, '- log_loss:', score1)
    total_score = log_loss(y, pred_oob)/6
    total_score1 = multi_log_loss(y, pred_oob)
    print('Total - log_loss:', total_score)
    print('Total - log_loss:', total_score1)
    ## test predictions
    pred_test /= (nfolds * nbags)
    submit(pred_test, total_score, kernel_name)
