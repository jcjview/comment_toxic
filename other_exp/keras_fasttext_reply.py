'''
The code is tested on Keras 2.0.0 using Theano backend, and Python 3.5
referrence Code:https://github.com/keras-team/keras/blob/master/examples/imdb_cnn_lstm.py
'''

import numpy as np
import pandas as pd
# from keras import initializations
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Input, Embedding, Dropout, RepeatVector, Conv1D, MaxPooling1D, GRU
from keras.layers import recurrent
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical, np_utils

MAX_FEATURES=10000
MAX_TEXT_LENGTH=20
EMBEDDING_DIM=300
TRAIN_DATA_FILE='./1.csv'
labels_index = {}
id_label={}

VALIDATION_SPLIT=0.1
kernel_name='emotion_hotel'
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
embed_path='char2vec_128.txt'
def get_model(word_index,class_num):
    # embedding_matrix=get_embedding_matrix(word_index,embed_path,EMBEDDING_DIM)
    nb_words = min(MAX_FEATURES, len(word_index))+1
    input1 = Input(shape=(MAX_TEXT_LENGTH,), dtype='int32')
    main = Embedding(MAX_FEATURES,
                     EMBEDDING_DIM,
                    input_length = MAX_TEXT_LENGTH,
                   # weights = [embedding_matrix],
                     )(input1)
    main = Dropout(0.2)(main)
    main = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(main)
    main = MaxPooling1D(pool_size=2)(main)
    main = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(main)
    main = MaxPooling1D(pool_size=2)(main)
    main = GRU(32)(main)
    main = Dense(16, activation="relu")(main)
    out = Dense(class_num, activation='softmax')(main)
    model = Model(inputs=input1, outputs=out)
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
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True,verbose=1,  save_weights_only=True,monitor='val_acc')

    hist = model.fit(data_train, labels_train,
                     validation_data=(data_val, labels_val),
                     epochs=50, batch_size=32,verbose=1,
                     callbacks=[early_stopping, model_checkpoint])

    model.load_weights(bst_model_path)
    bst_val_score = max(hist.history['val_acc'])
    print("bst_val_score",bst_val_score)
    ## make the submission
    print('Start making the submission before fine-tuning')
    # y_test = model.predict(test_data, batch_size=1024, verbose=1)
    # return  y_test,bst_val_score,STAMP

def predict(text_list,tokenizer,model):
    x_tokenized = tokenizer.texts_to_sequences(text_list)
    x=pad_sequences(x_tokenized, maxlen=MAX_TEXT_LENGTH)
    y_proba = model.predict(x)
    if y_proba.shape[-1] > 1:
        y_classes_id=y_proba.argmax(axis=-1)
    else:
        y_classes_id=(y_proba > 0.5).astype('int32')
    y_class_label=[]
    # y_classes_id = np_utils.to_categorical(y_proba)
    y_class_label=[id_label[l] for l in y_classes_id ]
    return y_class_label


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
    for label,id in labels_index.items():
        id_label[id]=label
    return y,class_num

def get_X_train_X_test(train):
    train_raw_text=train['text'].fillna("NA").values
    tokenizer = Tokenizer(num_words=MAX_FEATURES,filters='',lower=True)
    tokenizer.fit_on_texts(list(train_raw_text))
    train_tokenized = tokenizer.texts_to_sequences(train_raw_text)
    return pad_sequences(train_tokenized, maxlen=MAX_TEXT_LENGTH),tokenizer

def get_embedding_matrix(word_index,Emed_path,embedding_dims):
    # embedding_dims=EMBEDDING_DIM
    #    Glove Vectors
    print('Indexing word vectors')
    embeddings_index = {}
    f = open(Emed_path,encoding='utf-8')
    f.readline()
    for line in f:
        values = line.split()
        word = ' '.join(values[:-embedding_dims])
        coefs = np.asarray(values[-embedding_dims:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Total %s word vectors.' % len(embeddings_index))
    print('Preparing embedding matrix')
    nb_words = min(MAX_FEATURES, len(word_index))+1
    embedding_matrix = np.zeros((nb_words, embedding_dims))
    for word, i in word_index.items():
        if i >= MAX_FEATURES:
            continue
        # print(i,nb_words)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    # np.save(Emed_path, embedding_matrix)
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    return embedding_matrix


train = pd.read_csv(TRAIN_DATA_FILE)
X_train,tokenizer = get_X_train_X_test(train)
y,class_num=get_Y(train)
print('class num:', class_num)
print('word len',len(tokenizer.word_index))
print(X_train.shape)
print(y.shape)
model=get_model(tokenizer.word_index,class_num)
train_fit_predict(model, X_train, y)
# model.load_weights('emotion_0.25_0.25.h5')
# i=0
# while i<10000:
#     name =input("What's your name? ")
#     text_list=[]
#     t=" ".join([i for i in name])
#     text_list.append(t)
#     ou=predict(text_list,tokenizer,model)
#     print(ou)