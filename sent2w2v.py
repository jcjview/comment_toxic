import gensim
import h5py
import numpy as np
import pandas as pd
from keras.preprocessing import text, sequence

MAX_FEATURES = 20000
MAX_TEXT_LENGTH = 100
BATCH_SIZE = 32
EPOCHS = 10
VALIDATION_SPLIT = 0.1
embed_dim = 300
CLASSES_LIST = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

w2v_model =gensim.models.KeyedVectors.load_word2vec_format(fname='./data/GoogleNews-vectors-negative300.bin', binary=True)

train_rows = 95851
test_rows = 226998


def get_token_2_w2v(tokenized):
    # convert the wv word vectors into a numpy matrix that is suitable for insertion
    # into our TensorFlow and Keras models
    embedding_matrix = np.random.rand(MAX_TEXT_LENGTH, embed_dim)
    for i in range(len(tokenized)):
        embedding_vector = w2v_model.wv[tokenized[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


def get_word_vector_matrix(train):
    train_raw_text = train["comment_text"].fillna("MISSINGVALUE").values
    tokenizer = text.Tokenizer(num_words=MAX_FEATURES)
    tokenizer.fit_on_texts(list(train_raw_text))
    train_tokenized = tokenizer.texts_to_sequences(train_raw_text)
    return sequence.pad_sequences(train_tokenized, maxlen=MAX_TEXT_LENGTH)
def get_Y(train):
    return train[CLASSES_LIST].values

train = pd.read_csv("./data/train.csv")

X_train = get_word_vector_matrix(train)

outh5file = h5py.File('./data/train_w2v.h5', 'w')
train_d2v = outh5file.create_dataset('train_d2v', (train_rows, embed_dim), dtype=np.float32)
train_d2v[:] = get_token_2_w2v(X_train)[0:train_rows, :]
outh5file.flush()
outh5file.close()
print("OK")


test = pd.read_csv("./data/test.csv")
X_test = get_word_vector_matrix(test)
outh5file = h5py.File('./data/test_w2v.h5', 'w')
test_d2v = outh5file.create_dataset('test_d2v', (test_rows, embed_dim), dtype=np.float32)
test_d2v[:] = get_token_2_w2v(X_test)[0:test_rows, :]
outh5file.flush()
outh5file.close()
print("OK")


outh5file=h5py.File('./data/train_label.h5','w')
train_label_size = outh5file.create_dataset('train_label_size', (train_rows,len(CLASSES_LIST)),dtype=np.int8)
train_label_size[:]=get_Y(train)[0:train_rows, :]
outh5file.flush()
outh5file.close()
print("OK")