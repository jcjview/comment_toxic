import h5py
import numpy as np
import pandas as pd
import tqdm
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

import util.preprocessing as preprocessing
from config import *


def get_Y(train):
    return train[CLASSES_LIST].values


embedding_matrix_path = 'temp.npy'


def get_X_train_X_test(train_df, test_df, toxic1, toxic2):
    print('Processing text dataset')
    list_sentences_train = train_df["comment_text"].fillna("NA").values
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y = train_df[list_classes].values
    list_sentences_test = test_df["comment_text"].fillna("NA").values

    list_sentences_toxic1 = toxic1["comment"].fillna("NA").values
    list_sentences_toxic2 = toxic2["comment"].fillna("NA").values

    comments = []
    for text in list_sentences_train:
        comments.append(preprocessing.text_to_wordlist(text))
    test_comments = []
    for text in list_sentences_test:
        test_comments.append(preprocessing.text_to_wordlist(text))

    comments1 = []
    for text in list_sentences_toxic1:
        comments1.append(preprocessing.text_to_wordlist(text))

    comments2 = []
    for text in list_sentences_toxic2:
        comments2.append(preprocessing.text_to_wordlist(text))
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

    sequences1 = tokenizer.texts_to_sequences(comments1)
    sequences2 = tokenizer.texts_to_sequences(comments2)
    data1 = pad_sequences(sequences1, maxlen=MAX_TEXT_LENGTH)
    data2 = pad_sequences(sequences2, maxlen=MAX_TEXT_LENGTH)

    comments_new = comments + test_comments + comments1 + comments2

    with open('w2vcorpus.txt', 'w', encoding='utf-8') as thefile:
        for item in tqdm.tqdm(comments_new):
            thefile.write("%s\n" % item)

    return data, test_data, data1, data2, tokenizer.word_index


def load_train_test_y():
    print('load from h5 ', TRAIN_HDF5)
    outh5file = h5py.File(TRAIN_HDF5, 'r')
    train1 = outh5file['toxic1_token']
    train2 = outh5file['toxic2_token']
    y1 = outh5file['toxic1_label']
    y2 = outh5file['toxic2_label']
    return train1, train2, y1, y2


if __name__ == '__main__':
    train = pd.read_csv(TRAIN_DATA_FILE)
    test = pd.read_csv(TEST_DATA_FILE)

    toxic1 = pd.read_csv('../experiment2/train_toxicity.csv')
    toxic2 = pd.read_csv('../experiment2/attack_aggression.csv')

    X_train, X_test, data1, data2, word_index = get_X_train_X_test(train, test, toxic1, toxic2)
    class_list1 = ['toxicity']
    class_list2 = ['attack', 'aggression']
    y1 = toxic1[class_list1].values
    y2 = toxic2[class_list2].values
    TRAIN_HDF5 = 'toxic_pretrain.h5'
    print('save to h5 ', TRAIN_HDF5)
    outh5file = h5py.File(TRAIN_HDF5, 'w')
    toxic1_token = outh5file.create_dataset('toxic1_token', data=data1)
    toxic2_token = outh5file.create_dataset('toxic2_token', data=data2)
    toxic1_label = outh5file.create_dataset(name='toxic1_label', data=y1)
    toxic2_label = outh5file.create_dataset(name='toxic2_label', data=y2)
    outh5file.close()

    X_train1, X_train2, y_1, y_2 = load_train_test_y()
    embedding_matrix1 = np.load(embedding_matrix_path)
    print(X_train1.shape)
    print(X_train2.shape)
    print(y_1.shape)
    print(y_2.shape)
