#! -*- coding: utf-8 -*-
import mmap

import pandas as pd
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import *
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from sklearn.metrics import roc_auc_score
# 准备训练数据
from tqdm import tqdm

embedding_dims = 300
MAX_FEATURES = 72039
MAX_TEXT_LENGTH = 100
BATCH_SIZE = 32
EPOCHS = 10
VALIDATION_SPLIT = 0.1
SPLIT = 10000
CLASSES_LIST = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
path = './'

TRAIN_DATA_FILE = path + 'train.csv'
TEST_DATA_FILE = path + 'test.csv'

embedding_path = path + 'crawl-300d-2M.vec'
batch_size = 128
gru_len = 128
Routings = 5
Num_capsule = 10
Dim_capsule = 16
dropout_p = 0.3


def get_Y(train):
    return train[CLASSES_LIST].values


def predict(model, test_data, bst_model_path):
    model.load_weights(bst_model_path)
    y_test = model.predict(test_data, batch_size=1024, verbose=1)
    return y_test


def submit(y_test, bst_val_score, STAMP):
    sample_submission = pd.read_csv(path + "sample_submission.csv")
    sample_submission[CLASSES_LIST] = y_test
    sample_submission.to_csv('%.4f_' % (bst_val_score) + STAMP + '.csv', index=False)


def get_num_lines(file_path):
    with open(file_path, "r+") as fp:
        buf = mmap.mmap(fp.fileno(), 0)
        lines = 0
        while buf.readline():
            lines += 1
        return lines


def get_embedding_matrix(word_index, Emed_path):
    #    Glove Vectors
    print('Indexing word vectors')
    embeddings_index = {}
    file_line = get_num_lines(Emed_path)
    print('lines ', file_line)
    with open(Emed_path, encoding='utf-8') as f:
        for line in tqdm(f, total=file_line):
            values = line.split()
            word = ' '.join(values[:-300])
            coefs = np.asarray(values[-300:], dtype='float32')
            embeddings_index[word] = coefs
    f.close()

    print('Total %s word vectors.' % len(embeddings_index))
    print('Preparing embedding matrix')
    nb_words = min(MAX_FEATURES, len(word_index))
    all_embs = np.stack(embeddings_index.values())
    print(all_embs.shape)
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embedding_matrix = np.random.normal(loc=emb_mean, scale=emb_std, size=(nb_words, embedding_dims))
    # embedding_matrix = np.zeros((nb_words, embedding_dims))
    for word, i in tqdm(word_index.items()):
        if i >= MAX_FEATURES:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    # np.save(Emed_path, embedding_matrix)
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    return embedding_matrix


def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.

    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]

    text = " ".join(text)
    return (text)


def get_X_train_X_test(train_df, test_df):
    print('Processing text dataset')
    list_sentences_train = train_df["comment_text"].fillna("NA").values
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y = train_df[list_classes].values
    list_sentences_test = test_df["comment_text"].fillna("NA").values
    comments = []
    for text in tqdm(list_sentences_train):
        comments.append(text_to_wordlist(text))
    test_comments = []
    print('Processing test dataset')
    for text in tqdm(list_sentences_test):
        test_comments.append(text_to_wordlist(text))

    tokenizer = Tokenizer(num_words=MAX_FEATURES)
    tokenizer.fit_on_texts(comments + test_comments)

    sequences = tokenizer.texts_to_sequences(comments)
    test_sequences = tokenizer.texts_to_sequences(test_comments)

    word_index = tokenizer.word_index
    print('Found %s unique tokens' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_TEXT_LENGTH)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', y.shape)

    test_data = pad_sequences(test_sequences, maxlen=MAX_TEXT_LENGTH)
    print('Shape of test_data tensor:', test_data.shape)
    return data, test_data, tokenizer.word_index


def get_model(embedding_matrix):
    input1 = Input(shape=(MAX_TEXT_LENGTH,))
    embed_layer = Embedding(MAX_FEATURES,
                            embedding_dims,
                            input_length=MAX_TEXT_LENGTH,
                            weights=[embedding_matrix],
                            trainable=False)(input1)
    x = Bidirectional(
        GRU(gru_len, activation='relu', dropout=dropout_p, recurrent_dropout=dropout_p, return_sequences=True))(
        embed_layer)
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,
                      share_weights=True)(x)
    # output_capsule = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(capsule)
    capsule = Flatten()(capsule)
    capsule = Dropout(dropout_p)(capsule)
    output = Dense(6, activation='sigmoid')(capsule)
    model = Model(inputs=input1, outputs=output)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    model.summary()
    return model


def squash(x, axis=-1):
    # s_squared_norm is really small
    # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    # return scale * x
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale


# A Capsule Implement with Pure Keras
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

def predict(model,test_data,bst_model_path):
    model.load_weights(bst_model_path)
    y_test = model.predict(test_data, batch_size=1024, verbose=1)
    return y_test

train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)
data, X_test, word_index = get_X_train_X_test(train, test)
y = get_Y(train)
embedding_matrix1 = get_embedding_matrix(word_index, embedding_path)
model = get_model(embedding_matrix1)

x_train = data[:-SPLIT]
y_train = y[:-SPLIT]
print(x_train.shape, y_train.shape)

x_test = data[-SPLIT:]
y_test = y[-SPLIT:]
print(x_test.shape, y_test.shape)

bst_model_path = 'capsule' +  '.h5'
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, verbose=1, save_weights_only=True)
hist = model.fit(x_train, y_train,
                 batch_size=batch_size,
                 epochs=10,
                 verbose=1,
                 validation_data=(x_test, y_test),
                 callbacks=[early_stopping, model_checkpoint])

bst_val_score = min(hist.history['val_loss'])
print('val_loss',bst_val_score)
pred_val=predict(model,y_test,bst_model_path)
total_val_score = roc_auc_score(y, pred_val)
print('roc_auc_score',total_val_score)
pred_test=predict(model,X_test,bst_model_path)
submit(y_test,total_val_score,'capsule')

