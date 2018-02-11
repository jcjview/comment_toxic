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
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Conv1D, MaxPooling1D, Flatten, GlobalMaxPooling1D, \
    recurrent, RepeatVector
from keras.models import Model
from sklearn.metrics import roc_auc_score
from config import *
from util import preprocessing


class RocAucMetricCallback(Callback):
    def __init__(self, predict_batch_size=1024, include_on_batch=False):
        super(RocAucMetricCallback, self).__init__()
        self.predict_batch_size = predict_batch_size
        self.include_on_batch = include_on_batch

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        if (self.include_on_batch):
            logs['roc_auc_val'] = float('-inf')
            if (self.validation_data):
                logs['roc_auc_val'] = roc_auc_score(self.validation_data[1],
                                                    self.model.predict(self.validation_data[0],
                                                                       batch_size=self.predict_batch_size))

    def on_train_begin(self, logs={}):
        if not ('roc_auc_val' in self.params['metrics']):
            self.params['metrics'].append('roc_auc_val')

    def on_train_end(self, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        logs['roc_auc_val'] = float('-inf')
        if (self.validation_data):
            logs['roc_auc_val'] = roc_auc_score(self.validation_data[1],
                                                self.model.predict(self.validation_data[0],
                                                                   batch_size=self.predict_batch_size))

        ########################################
## set directories and parameters
########################################


rate_drop_dense=0.25
dense_size=64
kernel_size=10
cnn_filters=32
kernel_name='gru1.0'
num_lstm=64
rate_drop_lstm=0.25
lstm_output_size = 64
"""
 define the model structure

 """
def get_model(embedding_matrix):
    input1 = Input(shape=(MAX_TEXT_LENGTH,), dtype='int32')
    # x = Dropout(rate_drop_dense)(input1)
    embedding_layer = Embedding(MAX_FEATURES,
                                embedding_dims,
                                input_length=MAX_TEXT_LENGTH,
                                weights=[embedding_matrix],
                                trainable=False)
    x = embedding_layer(input1)
    # x = Dropout(rate_drop_dense)(x)
    x = recurrent.GRU(lstm_output_size,recurrent_dropout=rate_drop_lstm,dropout=rate_drop_dense)(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(dense_size, activation="relu")(x)
    x = Dropout(rate_drop_dense)(x)
    # x = RepeatVector(MAX_TEXT_LENGTH)(x)
    # x = recurrent.GRU(lstm_output_size,dropout=rate_drop_dense)(x)
    out = Dense(6, activation='sigmoid')(x)
    model = Model(inputs=input1, outputs=out)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['mse'])
    model.summary()
    model_json = model.to_json()
    with open("model_2gru.json", "w") as json_file:
        json_file.write(model_json)
    return model
"""
train the model
"""
def train_fit_predict(model, data,test_data,y):
    ########################################
    ## sample train/validation data
    ########################################
    # np.random.seed(1234)
    seed = 1
    np.random.seed(seed)
    cv_folds=10
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=cv_folds, random_state=seed, shuffle=False)
    y_cv = np.sum(y, axis=1)
    X_train = np.array(data, copy=True)
    y = np.array(y, copy=True)
    data_train=data_val=X_train
    labels_train=labels_val=y
    for ind_tr, ind_te in skf.split(X_train, y_cv):
        data_train = X_train[ind_tr]
        labels_train = y[ind_tr]

        data_val=X_train[ind_te]
        labels_val = y[ind_te]
        break
    print(data_train.shape, labels_train.shape)
    print(data_val.shape, labels_val.shape)

    STAMP =kernel_name+ '_%d_%.2f' % (dense_size, rate_drop_dense)
    print(STAMP)
    early_stopping = EarlyStopping(monitor='roc_auc_val', patience=5,mode='max')
    bst_model_path = STAMP + '.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path,monitor= 'roc_auc_val',mode='max',
                                       save_best_only=True,verbose=1,  save_weights_only=True)

    hist = model.fit(data_train, labels_train,
                     validation_data=(data_val, labels_val),
                     epochs=50, batch_size=256, shuffle=True,
                     callbacks=[RocAucMetricCallback(),early_stopping, model_checkpoint])

    model.load_weights(bst_model_path)
    bst_val_score = max(hist.history['roc_auc_val'])
    print("bst_roc_score",bst_val_score)
    ## make the submission
    print('Start making the submission before fine-tuning')
    y_test = model.predict(test_data, batch_size=1024, verbose=1)
    return  y_test,bst_val_score,STAMP
def submit(y_test,bst_val_score,STAMP):
    sample_submission = pd.read_csv(path+"sample_submission.csv")
    sample_submission[CLASSES_LIST] = y_test
    sample_submission.to_csv('%.4f_' % (bst_val_score) + STAMP + '.csv', index=False)

def get_Y(train):
    return train['toxicity'].values

embedding_matrix_path = 'temp.npy'

X_train, X_test, y, word_index = preprocessing.load_train_test_y()
embedding_matrix1 = np.load(embedding_matrix_path)
print(X_train.shape)
print(X_test.shape)
print(y.shape)
print(len(word_index))
print(embedding_matrix1.shape)

model = get_model(embedding_matrix1)
y_test, bst_val_score, STAMP = train_fit_predict(model, X_train, X_test, y)
submit(y_test, bst_val_score, STAMP)
