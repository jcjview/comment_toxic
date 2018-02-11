'''
The code is tested on Keras 2.0.0 using Theano backend, and Python 3.5
referrence Code:https://github.com/keras-team/keras/blob/master/examples/imdb_cnn_lstm.py
'''

import numpy as np
import pandas as pd
# from keras import initializations
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Input, Embedding, Dropout, RepeatVector, Conv1D, MaxPooling1D, GRU, Bidirectional
from keras.layers import recurrent
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

MAX_FEATURES=10000
MAX_TEXT_LENGTH=20
EMBEDDING_DIM=300
TRAIN_DATA_FILE='trains_zh.csv'
labels_index = {}
VALIDATION_SPLIT=0.1
kernel_name='emotion'
num_lstm = 300
num_dense = 256
rate_drop_lstm = 0.25
rate_drop_dense = 0.25

act = 'relu'

# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# LSTM
lstm_output_size = 70

def get_model(word_index,class_num):
    # embedding_matrix,nb_words=get_embedding_matrix(word_index)
    nb_words = min(MAX_FEATURES, len(word_index))
    input_layer = Input(shape=(MAX_TEXT_LENGTH,))
    embedding_layer = Embedding(MAX_FEATURES,
                                EMBEDDING_DIM,
                                input_length=MAX_TEXT_LENGTH,
                                # weights=[embedding_matrix],
                                trainable=False)(input_layer)
    x = Bidirectional(GRU(lstm_output_size, return_sequences=True))(embedding_layer)
    x = Dropout(rate_drop_dense)(x)
    x = Bidirectional(GRU(lstm_output_size, return_sequences=False))(x)
    x = Dense(64, activation="relu")(x)
    out = Dense(class_num, activation='sigmoid')(x)
    model = Model(inputs=input_layer, outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.summary()
    return model
"""
train the model
"""
def train_fit_predict(model, data, y):
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

    STAMP = kernel_name+'_%.2f_%.2f' % (rate_drop_lstm, rate_drop_dense)
    print(STAMP)
    early_stopping = EarlyStopping(monitor='val_acc', patience=5)
    bst_model_path = STAMP + '.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True,verbose=1,  save_weights_only=True)

    hist = model.fit(data_train, labels_train,
                     validation_data=(data_val, labels_val),
                     epochs=50, batch_size=256,verbose=1,
                     callbacks=[early_stopping, model_checkpoint])

    model.load_weights(bst_model_path)
    bst_val_score = min(hist.history['val_loss'])
    print("bst_val_score",bst_val_score)
    ## make the submission
    print('Start making the submission before fine-tuning')
    # y_test = model.predict(test_data, batch_size=1024, verbose=1)
    # return  y_test,bst_val_score,STAMP



def get_Y(train):
    labels = train['class'].values
    y=[]
    for l in labels:
        if l not in labels_index:
            label_id=len(labels_index)
            labels_index[l]=label_id
        y.append(labels_index[l])
    y = to_categorical(np.asarray(y))
    class_num=len(labels_index)

    return y,class_num

def get_X_train_X_test(train):
    train_raw_text=train['text']
    tokenizer = Tokenizer(num_words=MAX_FEATURES,filters='')
    tokenizer.fit_on_texts(list(train_raw_text))
    train_tokenized = tokenizer.texts_to_sequences(train_raw_text)
    return pad_sequences(train_tokenized, maxlen=MAX_TEXT_LENGTH),tokenizer.word_index

train = pd.read_csv(TRAIN_DATA_FILE)
X_train,word_index = get_X_train_X_test(train)
y,class_num=get_Y(train)
print('class num:', class_num)
print('word len',len(word_index))
print(X_train.shape)
print(y.shape)
model=get_model(word_index,class_num)
train_fit_predict(model, X_train, y)