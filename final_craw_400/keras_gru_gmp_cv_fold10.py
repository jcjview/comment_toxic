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
from keras.layers import Dense, Input, Embedding, Dropout, GlobalMaxPooling1D, \
    recurrent, GlobalAveragePooling1D, concatenate
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


rate_drop_dense = 0.28
dense_size = 64
kernel_size = 8
cnn_filters = 64
kernel_name = 'gru_gmp_gav_cv1'
num_lstm = 64
rate_drop_lstm = 0.25
lstm_output_size = 64

"""
 define the model structure

 """


def get_model(embedding_matrix):
    input1 = Input(shape=(MAX_TEXT_LENGTH,), dtype='int32')
    # x = Dropout(rate_drop_dense)(input1)
    embedding_layer = Embedding(MAX_FEATURES, embedding_dims, input_length=MAX_TEXT_LENGTH, weights=[embedding_matrix],
                                trainable=False)
    x = embedding_layer(input1)
    # x = Dropout(rate_drop_dense)(x)
    x = recurrent.GRU(lstm_output_size, recurrent_dropout=rate_drop_lstm, dropout=rate_drop_dense,
                      return_sequences=True)(x)
    max_pool = GlobalMaxPooling1D()(x)
    avg_pool = GlobalAveragePooling1D()(x)
    x = concatenate([avg_pool, max_pool])
    x = Dropout(rate_drop_dense)(x)
    x = Dense(dense_size, activation="relu")(x)
    # x = RepeatVector(MAX_TEXT_LENGTH)(x)
    # x = recurrent.GRU(lstm_output_size,dropout=rate_drop_dense)(x)
    out = Dense(6, activation='sigmoid')(x)
    model = Model(inputs=input1, outputs=out)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.summary()
    return model


"""
train the model
"""
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
    data_val1 = roll_matrix(data_val)
    data_train1 = roll_matrix(data_train)
    print(data_train.shape, labels_train.shape)
    print(data_val.shape, labels_val.shape)

    STAMP = kernel_name + '_%d_%.2f' % (bag, rate_drop_dense)
    print(STAMP)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')
    bst_model_path = STAMP + '.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, monitor='roc_auc_val', mode='max',
                                       save_best_only=True, verbose=1, save_weights_only=True)

    hist = model.fit(data_train, labels_train,
                     validation_data=(data_val, labels_val),
                     epochs=50, batch_size=256, shuffle=True,
                     callbacks=[RocAucMetricCallback(), early_stopping, model_checkpoint])
    bst_val_score = max(hist.history['roc_auc_val'])
    for i in range(3):
        hist1 = model.fit(data_train1, labels_train,
                         validation_data=(data_val1, labels_val),
                         epochs=1, batch_size=256, shuffle=True,
                         callbacks=[RocAucMetricCallback()])
        bst_val_score1 = max(hist1.history['roc_auc_val'])
        print(bst_val_score,bst_val_score1)
        if bst_val_score1<bst_val_score:
            continue
        else:
            model.save(bst_model_path)

    model.load_weights(bst_model_path)

    print("bst_val_score", bst_val_score)
    ## make the submission
    print('Start making the submission before fine-tuning')
    y_test = model.predict(test_data, batch_size=1024, verbose=1)
    return y_test, bst_val_score, STAMP


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
y_cv = np.sum(y, axis=1)
# skf = KFold(len(y), n_folds=cv_folds, shuffle=True, random_state=seed)
count = 0
pred_test = np.zeros(shape=(X_test.shape[0], 6))
pred_oob = np.zeros(y.shape)

X_train = np.array(X_train, copy=True)
y = np.array(y, copy=True)

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
# y_test, bst_val_score, STAMP = predict(model, X_train, X_test, y)

total_score = roc_auc_score(y, pred_oob)
print('Total - roc_auc_score:', total_score)
pred_test /= cv_folds
submit(pred_test, total_score, kernel_name)

tp = [0, 0, 0, 0, 0, 0]
fp = [0, 0, 0, 0, 0, 0]
fn = [0, 0, 0, 0, 0, 0]
tn = [0, 0, 0, 0, 0, 0]
tpall = 0
accbase = 0
recallbase = 0
y_pred = np.zeros(pred_oob.shape)
y_pred[pred_oob > 0.5] = 1
from sklearn.metrics import confusion_matrix

for i in range(6):
    tn[i], fp[i], fn[i], tp[i] = confusion_matrix(y[:, i], y_pred[:, i]).ravel()

    tpall += tp[i]
    accbase += fp[i] + tp[i]
    recallbase += fn[i] + tp[i]
acc = (tpall / accbase)
recall = (tpall / recallbase)
f1 = 2 * acc * recall / (acc + recall)
print('acc', acc)
print('recall', recall)
print('f1', f1)
error_vak_path = './log/' + kernel_name + '_error.txt'
with open(error_vak_path, 'w') as filepoint:
    for i in range(6):
        filepoint.write('{} {} {} {} {}\n'.format(CLASSES_LIST[i], tn[i], fp[i], fn[i], tp[i]))
    filepoint.write('roc_auc_score %.4f\n' % total_score)
    filepoint.write('acc %.4f\n' % acc)
    filepoint.write('recall %.4f\n' % recall)
    filepoint.write('f1 %.4f\n' % f1)
