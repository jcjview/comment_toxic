import pickle
import re

import h5py
import numpy as np
# from keras import initializations
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from config import *
from tqdm import tqdm
import mmap

def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

TRAIN_HDF5 = './train_hdf5.h5'
word_index_path='./word_index.pkl'



def dump_X_Y_train_test(train, test, y,word_index):
    print('save to h5 ',TRAIN_HDF5)
    outh5file = h5py.File(TRAIN_HDF5, 'w')
    test_token = outh5file.create_dataset('test_token', data=test)
    train_token = outh5file.create_dataset('train_token', data=train)
    train_label = outh5file.create_dataset(name='train_label', data=y)
    outh5file.close()
    with open(word_index_path, 'wb') as f:
        pickle.dump(word_index, f)

def load_train_test_y():
    print('load from h5 ',TRAIN_HDF5)
    outh5file = h5py.File(TRAIN_HDF5, 'r')
    train = outh5file['train_token']
    test = outh5file['test_token']
    y = outh5file['train_label']
    word_dict={}
    with open(word_index_path, 'rb') as f:
        word_dict= pickle.load(f)
    return train, test, y,word_dict

def get_X_train_X_test(train_df, test_df):
    print('Processing text dataset')
    list_sentences_train = train_df["comment_text"].fillna("NA").values
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y = train_df[list_classes].values
    list_sentences_test = test_df["comment_text"].fillna("NA").values

    with open('text.txt','w',encoding='utf-8') as fp:
         # for c in comments:
        fp.write("\n".join(list_sentences_train))
        fp.write("\n".join(list_sentences_test))
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

# Regex to remove all Non-Alpha Numeric and space
special_character_removal = re.compile(r'[^a-z\d ]', re.IGNORECASE)
# regex to replace all numerics
replace_numbers = re.compile(r'\d+', re.IGNORECASE)

def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.

    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]

    text = " ".join(text)

    # Remove Special Characters
    text = special_character_removal.sub('', text)

    # Replace Numbers
    text = replace_numbers.sub('n', text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return (text)


"""
prepare embeddings
"""


def get_embedding_matrix(word_index,Emed_path):
    #    Glove Vectors
    print('Indexing word vectors')
    embeddings_index = {}
    file_line=get_num_lines(Emed_path)
    print('lines ',file_line)
    with open(Emed_path,encoding='utf-8') as f:
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

