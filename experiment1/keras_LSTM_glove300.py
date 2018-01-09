import sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing import text
from keras.preprocessing import sequence
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model

from config import  *


def get_X_train_X_test(train, test):
    tokenizer = text.Tokenizer(num_words=MAX_FEATURES)
    tokenizer.fit_on_texts(train)
    train_tokenized = tokenizer.texts_to_sequences(train)
    test_tokenized = tokenizer.texts_to_sequences(test)
    return sequence.pad_sequences(train_tokenized, maxlen=MAX_TEXT_LENGTH), sequence.pad_sequences(test_tokenized, maxlen=MAX_TEXT_LENGTH)


def get_model():

    embedding_matrix = np.load("w2v_embedding_layer1.npz")["arr_0"]
    inp = Input(shape=(MAX_TEXT_LENGTH,))
    x = Embedding(MAX_FEATURES, embedding_dims, weights=[embedding_matrix])(inp)
    x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def train_fit_predict(model, X_train, X_test, y):
    print(X_train.shape)
    print(y.shape)
    print(X_test.shape)
    # validation_data
    valid_x = X_train[0:SPLIT]
    train_x = X_train[SPLIT:]
    valid_y = y[0:SPLIT]
    train_y = y[SPLIT:]
    file_path = "weights_LSTM_glove.best.hdf5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor="loss", mode="min", patience=5)
    callbacks_list = [checkpoint, early]
    model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1,
              validation_data=(valid_x, valid_y), callbacks=callbacks_list)
    model.load_weights(file_path)
    return model.predict(X_test)


def submit(y_test):
    sample_submission = pd.read_csv("./data/sample_submission.csv")
    sample_submission[CLASSES_LIST] = y_test
    sample_submission.to_csv("submission_keras_LSTM_glove300.csv", index=False)

train_corps = open(train_token_path,encoding='utf-8').readlines()
test = open(train_token_path,encoding='utf-8').readlines()
X_train,X_test = get_X_train_X_test(train_corps,test)
train = pd.read_csv(TRAIN_DATA_FILE)
y = train[CLASSES_LIST].values
y_test = train_fit_predict(get_model(), X_train, X_test, y)
submit(y_test)