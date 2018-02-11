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
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Conv1D, MaxPooling1D, Flatten, GlobalMaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from sklearn.cross_validation import StratifiedKFold, KFold
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
cnn_filters=64
kernel_name='CNN_cv'


"""
 define the model structure

 """
def get_model(embedding_matrix):
    comment_input = Input(shape=(MAX_TEXT_LENGTH,), dtype='int32')
    embedding_layer = Embedding(MAX_FEATURES,
                                embedding_dims,
                                weights=[embedding_matrix],
                                input_length=MAX_TEXT_LENGTH,
                                trainable=False)
    embedded_sequences = embedding_layer(comment_input)
    embedded_sequences=Dropout(rate_drop_dense)(embedded_sequences)
    main = Conv1D(filters=cnn_filters, kernel_size=kernel_size,
                  padding='same', activation='relu')(embedded_sequences)
    # main = MaxPooling1D(pool_size=6)(main)
    main=GlobalMaxPooling1D()(main)
    main = Flatten()(main)
    main = Dense(dense_size, activation="relu")(main)
    main=Dropout(rate_drop_dense)(main)
    preds = Dense(6, activation='sigmoid',name='logist')(main)
    model = Model(inputs=[comment_input],
                  outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model
"""
train the model
"""
def train_fit_predict(model, data_train,labels_train,data_val,labels_val,
                      test_data,bag):

    print(data_train.shape, labels_train.shape)
    print(data_val.shape, labels_val.shape)

    STAMP =kernel_name+ '_%d_%.2f' % (bag, rate_drop_dense)
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
    bst_val_score = min(hist.history['val_loss'])
    print("bst_val_score",bst_val_score)
    ## make the submission
    print('Start making the submission before fine-tuning')
    y_test = model.predict(test_data, batch_size=1024, verbose=1)
    return  y_test,bst_val_score,STAMP
def predict(model, data,test_data,y):
    ########################################
    ## sample train/validation data
    ########################################
    # np.random.seed(1234)

    data_train = data[:-SPLIT]
    labels_train = y[:-SPLIT]
    print(data_train.shape, labels_train.shape)

    data_val = data[-SPLIT:]
    labels_val = y[-SPLIT:]
    print(data_val.shape, labels_val.shape)

    STAMP =kernel_name+ '_%d_%.2f' % (dense_size, rate_drop_dense)
    print(STAMP)
    bst_model_path = STAMP + '.h5'
    # early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    # model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True,verbose=1,  save_weights_only=True)
    #
    # hist = model.fit(data_train, labels_train,
    #                  validation_data=(data_val, labels_val),
    #                  epochs=50, batch_size=256, shuffle=True,
    #                  callbacks=[early_stopping, model_checkpoint])

    model.load_weights(bst_model_path)
    # bst_val_score = min(hist.history['val_loss'])
    # print("bst_val_score",bst_val_score)
    ## make the submission
    bst_val_score=0.041111
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

seed = 1
np.random.seed(seed)
cv_folds=4
skf = KFold(len(y), n_folds=cv_folds, shuffle=True, random_state=seed)
count=0
pred_test = np.zeros(shape=(X_test.shape[0], 6))
pred_oob = np.zeros(y.shape)

X_train=np.array(X_train,copy=True)
y=np.array(y,copy=True)

for ind_tr, ind_te in skf:
    count+=1
    x_train = X_train[ind_tr]
    x_val = X_train[ind_te]

    y_train = y[ind_tr]
    y_val = y[ind_te]
    model = get_model(embedding_matrix1)
    y_test, bst_val_score, STAMP = train_fit_predict(model, x_train,y_train,x_val,y_val
                                                      ,X_test,count)
    y_val_pred=model.predict(x_val,batch_size=1024,verbose=1)
    pred_oob[ind_te]=y_val_pred
    pred_test+=y_test
# y_test, bst_val_score, STAMP = predict(model, X_train, X_test, y)

total_score = roc_auc_score(y, pred_oob)
print('Total - roc_auc_score:', total_score)
pred_test /= cv_folds
submit(pred_test, total_score, kernel_name)
