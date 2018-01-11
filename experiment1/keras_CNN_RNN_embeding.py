import numpy as np
import pandas as pd
from keras.models import Model
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
    tokenizer.fit_on_texts(list(train_raw_text))
    train_tokenized = tokenizer.texts_to_sequences(train_raw_text)
    test_tokenized = tokenizer.texts_to_sequences(test_raw_text)
    return sequence.pad_sequences(train_tokenized, maxlen=MAX_TEXT_LENGTH), sequence.pad_sequences(test_tokenized, maxlen=MAX_TEXT_LENGTH),tokenizer.word_index

def get_Y(train):
    return train[CLASSES_LIST].values

def get_model(word_index):
    embed_size = config.embedding_dims
    inp = Input(shape=(MAX_TEXT_LENGTH, ))
    embedding_layer = Embedding(MAX_FEATURES, embed_size)
    main = embedding_layer(inp)
    main = Dropout(0.2)(main)
    main = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(main)
    main = MaxPooling1D(pool_size=2)(main)
    main = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(main)
    main = MaxPooling1D(pool_size=2)(main)
    main = GRU(32)(main)
    main = Dense(16, activation="relu")(main)
    main = Dense(6, activation="sigmoid")(main)
    model = Model(inputs=inp, outputs=main)
    w1=get_embedding_matrix(word_index)
    embedding_layer.set_weights([w1])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def train_fit_predict(model, X_train, X_test, y):
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
    return model.predict(X_test)

def submit(y_test):
    sample_submission = pd.read_csv(config.path+"sample_submission.csv")
    sample_submission[CLASSES_LIST] = y_test
    sample_submission.to_csv("baseline_embeding.csv", index=False)

def get_embedding_matrix(word_index):
    embeddings_index={}
    def get_coefs(word,*arr):
        if word in word_index and word not in embeddings_index:
            try:
                vector=np.asarray(arr, dtype='float32')
                if len(arr)==embedding_dims:
                    embeddings_index[word]=vector
                else:
                    print(vector.shape)
            except:
                print('error ',word)
                #return word,np.random.normal(size=(embedding_dims,1))

    fp=open(GLOVE_EMBEDDING_FILE,encoding='utf-8')
    for o in fp:
        get_coefs(*o.strip().split())
    fp.close()
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
    return embedding_matrix
train = pd.read_csv(config.TRAIN_VALID_FILE)
test = pd.read_csv(config.TEST_DATA_FILE)

X_train, X_test,word_index = get_X_train_X_test(train, test)
y = get_Y(train)

y_test = train_fit_predict(get_model(word_index), X_train, X_test, y)

submit(y_test)