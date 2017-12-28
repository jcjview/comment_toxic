import h5py
import os
import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Input,Embedding
from keras.layers import GRU, Dropout, MaxPooling1D, Conv1D
from keras.models import Model
from config import  *



def get_model():
    inp = Input(shape=(MAX_TEXT_LENGTH,),dtype='int32')
    embedding_layer = Embedding(MAX_FEATURES, embedding_dims,dropout=0.2,
                    trainable=True)
    main=embedding_layer(inp)
    # main = Dropout(0.2)(main)
    main = Conv1D(filters=32, kernel_size=3, padding='valid', activation='relu')(main)
    main = MaxPooling1D(pool_size=2)(main)
    main = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(main)
    main = MaxPooling1D(pool_size=2)(main)
    main = GRU(32)(main)
    main = Dense(16, activation="relu")(main)
    main = Dense(6, activation="sigmoid")(main)
    model = Model(inputs=inp, outputs=main)

    if os.path.exists('w2v_embedding_layer.npz'):
        w1 = np.load("w2v_embedding_layer.npz")["arr_0"]
        embedding_layer.set_weights([w1])
        print("load embedding layer OK")
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def train_fit_predict(model, X_train, X_test, y):
    file_path = "weights_base.best.hdf5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor="loss", mode="min", patience=5)
    callbacks_list = [checkpoint, early]
    model.fit(X_train, y, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, shuffle=True,
              validation_split=VALIDATION_SPLIT, callbacks=callbacks_list)
    model.load_weights(file_path)
    return model.predict(X_test)


def submit(y_test):
    sample_submission = pd.read_csv("./data/sample_submission.csv")
    sample_submission[CLASSES_LIST] = y_test
    sample_submission.to_csv("baseline.csv", index=False)


train = h5py.File('./data/train_token.h5', 'r')
test = h5py.File('./data/test_token.h5', 'r')
X_train = train['train_token'][:]
X_test = test['test_token'][:]
train_label = h5py.File('./data/train_label.h5', 'r')
y = train_label['train_label'][:]

y_test = train_fit_predict(get_model(), X_train, X_test, y)

submit(y_test)
