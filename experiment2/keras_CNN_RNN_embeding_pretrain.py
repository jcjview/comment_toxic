import numpy as np
import pandas as pd
from keras.models import Model
from keras import Sequential
from keras.layers import Dense, Embedding, Input
from keras.layers import GRU, Dropout, MaxPooling1D, Conv1D
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import config
from config import *

file_path = "baseline_embeding_weights_base.best.hdf5"


def get_X_train_X_test(train, test,train_toxic):
    train_raw_text = train["comment_text"].fillna("NA").values
    test_raw_text = test["comment_text"].fillna("NA").values
    test_raw_toxic=train_toxic["comment"].fillna("NA").values
    tokenizer = text.Tokenizer(num_words=MAX_FEATURES)
    tokenizer.fit_on_texts(list(train_raw_text)+list(test_raw_text))
    train_tokenized = tokenizer.texts_to_sequences(test_raw_toxic)
    return sequence.pad_sequences(train_tokenized, maxlen=MAX_TEXT_LENGTH),tokenizer.word_index

def get_Y(train):
    return train['toxicity'].values

def get_model(word_index):
    if os.path.exists('w2v_embedding_layer.npz'):
        w1 = np.load("w2v_embedding_layer.npz")["arr_0"]
    else:
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
    logist=Dense(1, activation="sigmoid")
    main=embedding_layer(inp)
    main=dropout_layer(main)
    main=conv_layer1(main)
    main=maxpooling1(main)
    main=conv_layer2(main)
    main=maxpooling2(main)
    main=gru_layer(main)
    main=softmax(main)
    main=logist(main)
    model=Model(input=inp,outputs=main)
    # model = Sequential()
    # model.add(inp)
    # model.add(embedding_layer)
    # model.add(dropout_layer)
    # model.add(conv_layer1)
    # model.add(maxpooling1)
    # model.add(conv_layer2)
    # model.add(maxpooling2)
    # model.add(gru_layer)
    # model.add(softmax)
    # model.add(logist)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def train_fit_evaluate(model, X_train, y):
    test_x = X_train[0:SPLIT]
    test_y = y[0:SPLIT]
    valid_x=X_train[SPLIT:SPLIT2]
    valid_y=y[SPLIT:SPLIT2]
    train_x=X_train[SPLIT2:]
    train_y=y[SPLIT2:]
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    callbacks_list = [checkpoint, early]
    model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS,  verbose=1,
              validation_data=(valid_x, valid_y), callbacks=callbacks_list)
    model.evaluate(x=test_x,y=test_y,batch_size=1024,verbose=1)
    #model.load_weights(file_path)
    # model.fit(X_train, y, batch_size=BATCH_SIZE, epochs=EPOCHS,  shuffle=True,verbose=0)
    #return model.predict(X_test,batch_size=1024,verbose=1)
def get_embedding_matrix(word_index):
    def get_coefs(values):
        word = ''.join(values[:-300])
        coefs = np.asarray(values[-300:], dtype='float32')
        return word,coefs
    with open(GLOVE_EMBEDDING_FILE, encoding='utf-8') as fp:
        embeddings_index = dict(get_coefs(o.strip().split()) for o in fp)
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

train_pre=pd.read_csv(config.TRAIN_VALID_FILE)
toxic=pd.read_csv('train_toxicity.csv')

X_train,word_index = get_X_train_X_test(train, test,toxic)
y = get_Y(toxic)

train_fit_evaluate(get_model(word_index), X_train, y)

# submit(y_test)