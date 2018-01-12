'''
Single model may achieve LB scores at around 0.043
Don't need to be an expert of feature engineering
All you need is a GPU!!!!!!!

The code is tested on Keras 2.0.0 using Theano backend, and Python 3.5

referrence Code:https://www.kaggle.com/lystdo/lstm-with-word2vec-embeddings
'''

########################################
## import packages
########################################
import re

import numpy as np
import pandas as pd
from keras import backend as K
# from keras import initializations
from keras import initializers, regularizers, constraints
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.engine.topology import Layer
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


########################################
## set directories and parameters
########################################


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        # self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        # print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        # return input_shape[0], input_shape[-1]
        return input_shape[0], self.features_dim


import config

path = '../data/'
EMBEDDING_FILE = config.GLOVE_EMBEDDING_FILE
TRAIN_DATA_FILE = config.TRAIN_FILE
TEST_DATA_FILE = config.TEST_DATA_FILE

MAX_SEQUENCE_LENGTH = 150
MAX_NB_WORDS = 100000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

num_lstm = 300
num_dense = 256
rate_drop_lstm = 0.25
rate_drop_dense = 0.25

act = 'relu'

########################################
## index word vectors
########################################
print('Indexing word vectors')

# Glove Vectors
embeddings_index = {}
f = open(EMBEDDING_FILE)
for line in f:
    values = line.split()
    word = ' '.join(values[:-300])
    coefs = np.asarray(values[-300:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings_index))

train_df = pd.read_csv(TRAIN_DATA_FILE)
test_df = pd.read_csv(TEST_DATA_FILE)

########################################
## process texts in datasets
########################################
print('Processing text dataset')

# Regex to remove all Non-Alpha Numeric and space
special_character_removal = re.compile(r'[^a-z\d ]', re.IGNORECASE)

# regex to replace all numerics
replace_numbers = re.compile(r'\d+', re.IGNORECASE)


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

########################################
## prepare embeddings
########################################
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

print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

########################################
## sample train/validation data
########################################
# np.random.seed(1234)
perm = np.random.permutation(len(data))
idx_train = perm[:int(len(data) * (1 - VALIDATION_SPLIT))]
idx_val = perm[int(len(data) * (1 - VALIDATION_SPLIT)):]

data_train = data[idx_train]
labels_train = y[idx_train]
print(data_train.shape, labels_train.shape)

data_val = data[idx_val]
labels_val = y[idx_val]

print(data_val.shape, labels_val.shape)

########################################
## define the model structure
########################################
embedding_layer = Embedding(nb_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm, return_sequences=True)

comment_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(comment_input)
x = lstm_layer(embedded_sequences)
x = Dropout(rate_drop_dense)(x)
merged = Attention(MAX_SEQUENCE_LENGTH)(x)
merged = Dense(num_dense, activation=act)(merged)
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)
preds = Dense(6, activation='sigmoid')(merged)

########################################
## train the model
########################################
model = Model(inputs=[comment_input], \
              outputs=preds)
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
print(model.summary())

STAMP = 'simple_lstm_glove_vectors_%.2f_%.2f' % (rate_drop_lstm, rate_drop_dense)
print(STAMP)

early_stopping = EarlyStopping(monitor='val_loss', patience=5)
bst_model_path = STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

hist = model.fit(data_train, labels_train, \
                 validation_data=(data_val, labels_val), \
                 epochs=50, batch_size=256, shuffle=True, \
                 callbacks=[early_stopping, model_checkpoint])

model.load_weights(bst_model_path)
bst_val_score = min(hist.history['val_loss'])

#######################################
## make the submission
########################################
print('Start making the submission before fine-tuning')

y_test = model.predict([test_data], batch_size=1024, verbose=1)

sample_submission = pd.read_csv("../input/sample_submission.csv")
sample_submission[list_classes] = y_test

sample_submission.to_csv('%.4f_' % (bst_val_score) + STAMP + '.csv', index=False)
