'''
Single model may achieve LB scores at around 0.043
Don't need to be an expert of feature engineering
All you need is a GPU!!!!!!!

The code is tested on Keras 2.0.0 using Theano backend, and Python 3.5

referrence Code:https://www.kaggle.com/lystdo/lstm-with-word2vec-embeddings
'''

import numpy as np
import pandas as pd
# from keras import initializations
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Bidirectional, GlobalMaxPool1D, GRU
from keras.models import Model
from keras.optimizers import RMSprop

import config
from util import preprocessing

"""
 define the model structure

 """

rate_drop_lstm=0.25
rate_drop_dense=0.25
recurrent_units=64
dense_size=64
kernel_name='BGRU1.0'
def get_model(word_index,embedding_matrix):
    # embedding_matrix,nb_words=get_embedding_matrix(word_index)
    nb_words = min(config.MAX_FEATURES, len(word_index))
    input = Input(shape=(config.MAX_TEXT_LENGTH,), dtype='int32')

    x = Embedding(nb_words, config.embedding_dims, input_length=config.MAX_FEATURES,
                  weights=[embedding_matrix], trainable=False)(input)
    x = Bidirectional(GRU(recurrent_units, return_sequences=True))(x)
    x = Dropout(rate_drop_dense)(x)
    x = Bidirectional(GRU(recurrent_units, return_sequences=False))(x)
    x = Dense(dense_size, activation="relu")(x)
    output_layer = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=input, outputs=output_layer)
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(clipvalue=1, clipnorm=1),
                  metrics=['accuracy'])
    model.summary()
    return model

"""
train the model
"""


def train_fit_predict(model, data, test_data, y):
    ########################################
    ## sample train/validation data
    ########################################
    # np.random.seed(1234)
    data_train = data[config.SPLIT:]
    labels_train = y[config.SPLIT:]
    print(data_train.shape, labels_train.shape)

    data_val = data[:config.SPLIT]
    labels_val = y[:config.SPLIT]
    print(data_val.shape, labels_val.shape)

    STAMP = kernel_name+'_%.2f_%.2f' % (rate_drop_lstm, rate_drop_dense)
    print(STAMP)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    bst_model_path = STAMP + '.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, verbose=1, save_weights_only=True)

    hist = model.fit(data_train, labels_train,
                     validation_data=(data_val, labels_val),
                     epochs=50, batch_size=256, shuffle=True,
                     callbacks=[early_stopping, model_checkpoint])

    model.load_weights(bst_model_path)
    bst_val_score = min(hist.history['val_loss'])
    print("bst_val_score", bst_val_score)
    ## make the submission
    print('Start making the submission before fine-tuning')
    y_test = model.predict(test_data, batch_size=1024, verbose=1)
    return y_test, bst_val_score, STAMP


def submit(y_test, bst_val_score, STAMP):
    sample_submission = pd.read_csv(config.path + "sample_submission.csv")
    sample_submission[config.CLASSES_LIST] = y_test
    sample_submission.to_csv('%.4f_' % (bst_val_score) + STAMP + '.csv', index=False)


embedding_matrix_path = 'temp.npy'

X_train, X_test, y, word_index = preprocessing.load_train_test_y()
embedding_matrix1 = np.load(embedding_matrix_path)
print(X_train.shape)
print(X_test.shape)
print(y.shape)
print(len(word_index))
print(embedding_matrix1.shape)

model = get_model(word_index,embedding_matrix1)
y_test, bst_val_score, STAMP = train_fit_predict(model, X_train, X_test, y)
submit(y_test, bst_val_score, STAMP)
