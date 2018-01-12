import numpy as np
import pandas as pd
from keras.models import Model
from keras import Sequential
from keras.layers import Dense, Embedding, Input
from keras.layers import GRU, Dropout, MaxPooling1D, Conv1D
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint

import config
from config import *

file_path = "baseline_embeding_weights_base.best.hdf5"


def get_X_train_X_test(train, test):
    train_raw_text = train["comment_text"].fillna("MISSINGVALUE").values
    test_raw_text = test["comment_text"].fillna("MISSINGVALUE").values
    tokenizer = text.Tokenizer(num_words=MAX_FEATURES)
    tokenizer.fit_on_texts(list(train_raw_text)+list(test_raw_text))
    train_tokenized = tokenizer.texts_to_sequences(train_raw_text)
    test_tokenized = tokenizer.texts_to_sequences(test_raw_text)
    return sequence.pad_sequences(train_tokenized, maxlen=MAX_TEXT_LENGTH), sequence.pad_sequences(test_tokenized, maxlen=MAX_TEXT_LENGTH),tokenizer.word_index

def get_Y(train):
    return train[CLASSES_LIST].values

def get_model(word_index):
    w1 = get_embedding_matrix(word_index)
    embed_size = config.embedding_dims
    inp = Input(shape=(MAX_TEXT_LENGTH, ))
    embedding_layer = Embedding(MAX_FEATURES, embed_size,weights=[w1])
    dropout_layer=Dropout(0.2)
    conv_layer1=Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')
    maxpooling1= MaxPooling1D(pool_size=2)
    conv_layer2 = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')
    maxpooling2= MaxPooling1D(pool_size=2)
    gru_layer=GRU(32)
    softmax=Dense(16, activation="relu")
    logist6=Dense(6, activation="sigmoid")
    model = Sequential()
    model.add(inp)
    model.add(embedding_layer)
    model.add(dropout_layer)
    model.add(conv_layer1)
    model.add(maxpooling1)
    model.add(conv_layer2)
    model.add(maxpooling2)
    model.add(gru_layer)
    model.add(softmax)
    model.add(logist6)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def train_fit_predict(model, X_train, X_test, y):
    model.load_weights(file_path,by_name=True)
    valid_x=X_train[0:SPLIT]
    valid_y=y[0:SPLIT]
    train_x=X_train[SPLIT:]
    train_y=y[SPLIT:]
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=20)
    callbacks_list = [checkpoint, early]
    model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS,  verbose=1,
              validation_data=(valid_x, valid_y), callbacks=callbacks_list)

    model.load_weights(file_path)
    # model.fit(X_train, y, batch_size=BATCH_SIZE, epochs=EPOCHS,  shuffle=True,verbose=0)
    return model.predict(X_test)

def submit(y_test):
    sample_submission = pd.read_csv(config.path+"sample_submission.csv")
    sample_submission[CLASSES_LIST] = y_test
    sample_submission.to_csv("baseline_embeding.csv", index=False)

def get_embedding_matrix(word_index):
    def get_coefs(word, *arr):
        def get_coefs(values):
            word = ''.join(values[:-300])
            coefs = np.asarray(values[-300:], dtype='float32')
            return word, coefs
    with open(GLOVE_EMBEDDING_FILE, encoding='utf-8') as fp:
        embeddings_index = dict(get_coefs(*o.strip().split()) for o in fp)
    all_embs = np.stack(embeddings_index.values())
    print(all_embs.shape)
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    nb_words = min(MAX_FEATURES, len(word_index))
    embedding_matrix = np.random.normal(loc=emb_mean,scale=emb_std,size=(nb_words, embedding_dims))

    for word, i in word_index.items():
        if i >= MAX_FEATURES: continue
        if word in embeddings_index:
            embedding_matrix[i] = embeddings_index[word]
    print(len(embedding_matrix))
    np.savez("w2v_embedding_layer.npz", embedding_matrix)
    return embedding_matrix
train = pd.read_csv(config.TRAIN_VALID_FILE)
test = pd.read_csv(config.TEST_DATA_FILE)

X_train, X_test,word_index = get_X_train_X_test(train, test)
y = get_Y(train)

y_test = train_fit_predict(get_model(word_index), X_train, X_test, y)

submit(y_test)