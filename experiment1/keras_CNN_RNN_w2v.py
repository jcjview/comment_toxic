import h5py
import os
import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Input, Embedding, SpatialDropout1D
from keras.layers import GRU, Dropout, MaxPooling1D, Conv1D
from keras.preprocessing import text, sequence
from keras.models import Model
from config import  *
from keras.optimizers import SGD, Adam
from keras.layers import  regularizers
def get_X_train_X_test(train, test):
    print('trainset ',len(train))
    print('testset ',len(test))
    tokenizer = text.Tokenizer(num_words=MAX_FEATURES)
    tokenizer.fit_on_texts(train)
    train_tokenized = tokenizer.texts_to_sequences(train)
    test_tokenized = tokenizer.texts_to_sequences(test)
    return sequence.pad_sequences(train_tokenized, maxlen=MAX_TEXT_LENGTH), sequence.pad_sequences(test_tokenized, maxlen=MAX_TEXT_LENGTH)

def get_model():
    inp = Input(shape=(MAX_TEXT_LENGTH,),dtype='int32')
    embedding_layer = Embedding(MAX_FEATURES, embedding_dims,
                    trainable=True)
    main=embedding_layer(inp)
    # main = Dropout(0.2)(main)
    #main=SpatialDropout1D(0.2)(main)#Dropout
    main = Conv1D(filters=32, kernel_size=5, padding='valid', activation='relu')(main)
    main = MaxPooling1D(pool_size=2)(main)
    main = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(main)
    main = MaxPooling1D(pool_size=2)(main)
    main = GRU(32)(main)
    main = Dense(16, activation="relu")(main)
    main = Dense(6, activation="sigmoid")(main)
    #main = Dense(6, activation="sigmoid",kernel_regularizer=regularizers.l2(0.01))(main)

    model = Model(inputs=inp, outputs=main)
    w1 = np.load("w2v_embedding_layer1.npz")["arr_0"]
    embedding_layer.set_weights([w1])
    print("load embedding layer OK")
    #sgd = SGD(lr=0.01, momentum=0.8, decay=0.0, nesterov=False)
    adam=Adam(lr=0.002,decay=1e-6)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()
    return model


def train_fit_predict(model, X_train, X_test, y):
    print(X_train.shape)
    print(y.shape)
    print(X_test.shape)
    # validation_data
    valid_x=X_train[0:SPLIT]
    train_x=X_train[SPLIT:]
    valid_y=y[0:SPLIT]
    train_y=y[SPLIT:]
    file_path = "weights_base.best.hdf5"
    # model.load_weights(file_path)
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor="loss", mode="min", patience=5)
    callbacks_list = [checkpoint, early]
    #model.fit(X_train, y, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1,validation_split=VALIDATION_SPLIT, shuffle=True,callbacks=callbacks_list)
    model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1,validation_data=(valid_x,valid_y), shuffle=True,callbacks=callbacks_list)

    model.load_weights(file_path)
    #return model.predict(X_test)

def predict(model,X_train,X_test,y):
    print(X_train.shape)
    print(y.shape)
    print(X_test.shape)
    file_path = "weights_base.best.hdf5"
    model.load_weights(file_path)
    return model.predict(X_test, batch_size=1024, verbose=1)

def submit(y_test):
    sample_submission = pd.read_csv("../data/sample_submission.csv")
    sample_submission[CLASSES_LIST] = y_test
    sample_submission.to_csv("CNN_RNN_enhance.csv", index=False)


train = open(train_token_path,encoding='utf-8').readlines()
test = open(test_token_path,encoding='utf-8').readlines()
X_train,X_test = get_X_train_X_test(train,test)
train = pd.read_csv(TRAIN_VALID_FILE)
y = train[CLASSES_LIST].values

train_fit_predict(get_model(), X_train, X_test, y)
#y_test = predict(get_model(), X_train, X_test, y)

#submit(y_test)
