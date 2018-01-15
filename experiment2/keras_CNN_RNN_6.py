import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import GRU, Dropout, MaxPooling1D, Conv1D
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint

import config
from config import *

file_path = "_CNN_embeding_weights.h5"


def get_X_train_X_test(train, test):
    train_raw_text = train["comment_text"].fillna("MISSINGVALUE").values
    test_raw_text = test["comment_text"].fillna("MISSINGVALUE").values
    tokenizer = text.Tokenizer(num_words=MAX_FEATURES)
    tokenizer.fit_on_texts(list(train_raw_text)+list(test_raw_text))
    train_tokenized = tokenizer.texts_to_sequences(train_raw_text)
    test_tokenized = tokenizer.texts_to_sequences(test_raw_text)
    return sequence.pad_sequences(train_tokenized, maxlen=MAX_TEXT_LENGTH), sequence.pad_sequences(test_tokenized, maxlen=MAX_TEXT_LENGTH),tokenizer.word_index

def get_Y(train,label):
    return train[label].values

def get_model(w1):
    embed_size = config.embedding_dims
    inp = Input(shape=(MAX_TEXT_LENGTH, ))
    embedding_layer = Embedding(MAX_FEATURES, embed_size)
    embedding_layer.set_weights([w1])
    main = embedding_layer(inp)
    main = Dropout(0.2)(main)
    main = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(main)
    main = MaxPooling1D(pool_size=2)(main)
    main = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(main)
    main = MaxPooling1D(pool_size=2)(main)
    main = GRU(32)(main)
    main = Dense(16, activation="relu")(main)
    main = Dense(1, activation="sigmoid")(main)
    model = Model(inputs=inp, outputs=main)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def train_fit_predict(model, X_train, X_test, y,label):
    valid_x=X_train[0:SPLIT]
    valid_y=y[0:SPLIT]
    train_x=X_train[SPLIT:]
    train_y=y[SPLIT:]
    label_file_path=label+file_path
    checkpoint = ModelCheckpoint(label_file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=20)
    callbacks_list = [checkpoint, early]
    model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=10,  verbose=1,
              validation_data=(valid_x, valid_y), callbacks=callbacks_list)

    model.load_weights(label_file_path)
    # model.fit(X_train, y, batch_size=BATCH_SIZE, epochs=EPOCHS,  shuffle=True,verbose=0)
    return model.predict(X_test)

def submit(y_test):
    sample_submission = pd.read_csv(config.path+"sample_submission.csv")
    sample_submission[CLASSES_LIST] = y_test
    sample_submission.to_csv("baseline_embeding.csv", index=False)

def get_embedding_matrix(word_index):
    def get_coefs(word, *arr):
        try:
            vector=np.asarray(arr, dtype='float32')
            if len(arr)==embedding_dims:
                return word, vector
            else:
                print('error ', word)
                return word, np.random.normal(size=(embedding_dims, ))
        except:
            # print('error ', word)
            return word, np.random.normal(size=(embedding_dims, ))
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
    return embedding_matrix
train = pd.read_csv(config.TRAIN_VALID_FILE)
test = pd.read_csv(config.TEST_DATA_FILE)

X_train, X_test,word_index = get_X_train_X_test(train, test)
w1 = get_embedding_matrix(word_index)
y_test={}
for label in CLASSES_LIST:
    print('label ',label)
    y = get_Y(train,label)
    y_test[label] = train_fit_predict(get_model(w1), X_train, X_test, y,label)

submit(y_test)