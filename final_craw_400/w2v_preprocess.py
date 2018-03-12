import multiprocessing

from keras.preprocessing.text import Tokenizer

from config import *
import util.preprocessing as preprocessing
from util.preprocessing import text_to_wordlist
import pandas as pd
import numpy as np
def get_Y(train):
    return train[CLASSES_LIST].values
embedding_matrix_path='temp.npy'

def multi_preprocess(comments=[]):
    pool_size = multiprocessing.cpu_count()+1
    print("pool_size", pool_size)
    pool = multiprocessing .Pool(pool_size)
    pool_outputs = pool.map(text_to_wordlist, comments)
    pool.close()
    pool.join()
    print('successful')
    return pool_outputs

def get_X_train_X_test(train_df, test_df, toxic1, toxic2):
    print('Processing text dataset')
    list_sentences_train = train_df["comment_text"].fillna("NA").values
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y = train_df[list_classes].values
    list_sentences_test = test_df["comment_text"].fillna("NA").values

    list_sentences_toxic1 = toxic1["comment"].fillna("NA").values
    list_sentences_toxic2 = toxic2["comment"].fillna("NA").values

    comments = list(list_sentences_train)
    comments = multi_preprocess(comments)

    test_comments = list(list_sentences_test)
    test_comments = multi_preprocess(test_comments)

    comments1 =  list(list_sentences_toxic1)
    comments1 = multi_preprocess(comments1)

    comments2 = list(list_sentences_toxic2)
    comments2 = multi_preprocess(comments2)

     # tokenizer = Tokenizer(num_words=MAX_FEATURES)
    # tokenizer.fit_on_texts(comments + test_comments)
    #
    # sequences = tokenizer.texts_to_sequences(comments)
    # test_sequences = tokenizer.texts_to_sequences(test_comments)
    #
    # word_index = tokenizer.word_index
    # print('Found %s unique tokens' % len(word_index))

    #data = pad_sequences(sequences, maxlen=MAX_TEXT_LENGTH)
    #print('Shape of data tensor:', data.shape)
    #print('Shape of label tensor:', y.shape)

    #test_data = pad_sequences(test_sequences, maxlen=MAX_TEXT_LENGTH)
    #print('Shape of test_data tensor:', test_data.shape)

    #sequences1 = tokenizer.texts_to_sequences(comments1)
    #sequences2 = tokenizer.texts_to_sequences(comments2)
    #data1 = pad_sequences(sequences1, maxlen=MAX_TEXT_LENGTH)
    #data2 = pad_sequences(sequences2, maxlen=MAX_TEXT_LENGTH)

    comments_new = comments + test_comments + comments1 + comments2

    with open('w2vcorpus.txt', 'w', encoding='utf-8') as thefile:
        for item in comments_new:
            thefile.write("%s\n" % item)

    #return data, test_data, data1, data2, tokenizer.word_index

if __name__ == '__main__':
    train = pd.read_csv('../data/train_valid_test.csv')
    test = pd.read_csv(TEST_DATA_FILE)

    toxic1 = pd.read_csv('../experiment2/train_toxicity.csv')
    toxic2 = pd.read_csv('../experiment2/attack_aggression.csv')

    get_X_train_X_test(train, test, toxic1, toxic2)
    
    
    #train = pd.read_csv(TRAIN_DATA_FILE)
    #test = pd.read_csv(TEST_DATA_FILE)
    #X_train, X_test, word_index = preprocessing.get_X_train_X_test(train, test)
    #embedding_matrix = preprocessing.get_embedding_matrix(word_index,GLOVE_EMBEDDING_FILE)
    #np.save(embedding_matrix_path, embedding_matrix)
    #y = get_Y(train)

    #print('random')
    #perm = np.random.permutation(len(y))
    #X_train=X_train[perm]
    #y=y[perm]



    #preprocessing.dump_X_Y_train_test(X_train, X_test, y, word_index)
    #X_train, X_test, y, word_index=preprocessing.load_train_test_y()
    #embedding_matrix1=np.load(embedding_matrix_path)
    #print(X_train.shape)
    #print(X_test.shape)
    #print(y.shape)
    #print(len(word_index))
    #print(embedding_matrix1.shape)
    #with open('word_index.txt','w') as fp:
    #    for word,index in word_index.items():
    #        fp.write(word+"\n")
        