'''
Don't need to be an expert of feature engineering
All you need is a GPU!!!!!!!

The code is tested on Keras 2.0.0 using Theano backend, and Python 3.5
'''

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.layers import Dense, Input, Embedding, Dropout, Conv1D, GlobalMaxPooling1D, recurrent, RepeatVector, \
    Bidirectional, GRU, BatchNormalization, SpatialDropout1D, GlobalAveragePooling1D, concatenate
from keras.models import Model
from keras.optimizers import RMSprop, Adam
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


rate_drop_dense = 0.28
dense_size = 64
kernel_size = 9
cnn_filters = 64
lstm_output_size = 64
kernel_name = 'pooled_gru_cnn_cv_fold10'
"""
 define the model structure

 """
units = 128
dr = 0.2
lr = 1e-3
lr_d = 0



def get_model(embedding_matrix):
    inp = Input(shape=(MAX_TEXT_LENGTH,))
    x = Embedding(MAX_FEATURES, embedding_dims, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(dr)(x)
    x = Bidirectional(GRU(units, return_sequences=True))(x)
    x = Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy"])

    return model

"""
train the model
"""
#to save memory
# def generate_batch_data_random(x, y, batch_size):
#     """逐步提取batch数据到显存，降低对显存的占用"""
#     idx = np.arange(len(y))
#     np.random.shuffle(idx)
#     batches = [idx[range(batch_size*i, min(len(y), batch_size*(i+1)))] for i in range(len(y)//batch_size+1)]
#
#     while (True):
#         for i in batches:
#             yield np.roll(x[i], axis=1, shift=np.random.randint(x[i].shape[1])), y[i]

def roll_matrix(data_train):
    size = data_train.shape[0]
    batch_size = 32
    data = np.array(data_train, copy=True)
    for index in range(size // batch_size - 1):
        print(index * batch_size, (index+1) * batch_size)
        if (index+1) * batch_size > size:
            break
        i = [index * batch_size, (index+1) * batch_size]
        data[i] = np.roll(data_train[i], axis=1, shift=np.random.randint(data_train[i].shape[1]))
    return data
def train_fit_predict(model, data_train, labels_train, data_val, labels_val,
                      test_data, bag):

    # data_val=roll_matrix(data_val)
    # data_train=roll_matrix(data_train)
    print(data_train.shape, labels_train.shape)
    print(data_val.shape, labels_val.shape)
    STAMP = kernel_name + '_%d_%.2f' % (bag, rate_drop_dense)
    print(STAMP)
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, mode='min')
    bst_model_path = STAMP + '.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, monitor='val_loss', mode='min',
                                       save_best_only=True, verbose=1, save_weights_only=True)
    from keras.callbacks import CSVLogger

    csv_logger = CSVLogger('./log/'+STAMP+'log.csv', append=True, separator=';')
    batch_size=256
    # hist =model.fit_generator(generator =generate_batch_data_random(data_train, labels_train, batch_size),
    #                           steps_per_epoch=len(data_train) / batch_size,
    #                           epochs=50,
    #                           validation_data=(data_val, labels_val),
    #                           verbose=1,
    #                           callbacks=[RocAucMetricCallback(), early_stopping, model_checkpoint,csv_logger])

    hist = model.fit(data_train, labels_train,
                     validation_data=(data_val, labels_val),
                     epochs=12, batch_size=256, shuffle=True,
                     callbacks=[RocAucMetricCallback(), early_stopping, model_checkpoint,csv_logger])

    model.load_weights(bst_model_path)
    bst_roc_auc_val = min(hist.history['val_loss'])
    print("bst_val_score", bst_roc_auc_val)

    ## make the submission
    print('Start making the submission before fine-tuning')
    y_test = model.predict(test_data, batch_size=1024, verbose=1)
    return y_test, bst_roc_auc_val, STAMP


def submit(y_test, bst_val_score, STAMP):
    sample_submission = pd.read_csv(path + "sample_submission.csv")
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

seed = 1
np.random.seed(seed)
cv_folds = 10

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=cv_folds, random_state=seed, shuffle=False)
count = 0
pred_test = np.zeros(shape=(X_test.shape[0], 6))
pred_oob = np.zeros(y.shape)

y_cv = np.sum(y, axis=1)
print(y_cv.shape)

X_train = np.array(X_train, copy=True)
y = np.array(y, copy=True)
for ind_tr, ind_te in skf.split(X_train, y_cv):
    y_val = y[ind_te]
    print(np.sum(y_val))

for ind_tr, ind_te in skf.split(X_train, y_cv):
    count += 1
    x_train = X_train[ind_tr]
    x_val = X_train[ind_te]

    y_train = y[ind_tr]
    y_val = y[ind_te]
    model = get_model(embedding_matrix1)
    y_test, bst_val_score, STAMP = train_fit_predict(model, x_train, y_train, x_val, y_val
                                                     , X_test, count)
    y_val_pred = model.predict(x_val, batch_size=1024, verbose=1)
    pred_oob[ind_te] = y_val_pred
    pred_test += y_test

total_score = roc_auc_score(y, pred_oob)
print('Total - roc_auc_score:', total_score)
pred_test /= cv_folds
submit(pred_test, total_score, kernel_name)
