import multiprocessing

import pandas as pd
import pickle

from config import *
from util.preprocessing import text_to_wordlist


def get_Y(train):
    return train[CLASSES_LIST].values


embedding_matrix_path = 'temp.npy'


def multi_preprocess(comments=[]):
    pool_size = multiprocessing.cpu_count() + 1
    print("pool_size", pool_size)
    pool = multiprocessing.Pool(pool_size)
    pool_outputs = pool.map(text_to_wordlist, comments)
    pool.close()
    pool.join()
    print('successful')
    return pool_outputs


from sklearn.feature_extraction.text import CountVectorizer


def get_X_train_X_test(train_df, test_df, toxic1, toxic2):
    print('Processing text dataset')
    list_sentences_train = train_df["comment_text"].fillna("NA").values
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    # y = train_df[list_classes].values
    list_sentences_test = test_df["comment_text"].fillna("NA").values

    list_sentences_toxic1 = toxic1["comment"].fillna("NA").values
    list_sentences_toxic2 = toxic2["comment"].fillna("NA").values

    comments = list(list_sentences_train)
    comments = multi_preprocess(comments)

    test_comments = list(list_sentences_test)
    test_comments = multi_preprocess(test_comments)
    with open('comments.pkl', 'wb') as f:
        pickle.dump(comments, f)
    with open('testcomments.pkl', 'wb') as f:
        pickle.dump(test_comments, f)

    # comments1 =  list(list_sentences_toxic1)
    # comments1 = multi_preprocess(comments1)
    #
    # comments2 = list(list_sentences_toxic2)
    # comments2 = multi_preprocess(comments2)

    # with open('comments.pkl', 'rb') as f:
    #     comments = pickle.load(f)
    # with open('testcomments.pkl', 'rb') as f:
    #     test_comments = pickle.load(f)
    cv = CountVectorizer(max_features=MAX_FEATURES)
    comments_fit = cv.fit_transform(comments)
    name = cv.get_feature_names()
    count = comments_fit.toarray().sum(axis=0)
    comments_word_dict = {}
    with open('train_word.txt', 'w', encoding='utf-8') as thefile:
        for index, n in enumerate(name):
            comments_word_dict[n] = count[index]
            thefile.write("%s\t%d\n" % (n, count[index]))

    cv1 = CountVectorizer(max_features=MAX_FEATURES)
    testfit = cv1.fit_transform(test_comments)
    name = cv1.get_feature_names()
    count = testfit.toarray().sum(axis=0)
    test_word_dict = {}
    for index, n in enumerate(name):
        if n not in comments_word_dict:
            test_word_dict[n] = count[index]

    sortedlist = sorted(test_word_dict.items(), key=lambda d: d[1], reverse=True)
    with open('test_word.txt', 'w', encoding='utf-8') as thefile:
        for k, v in sortedlist:
            thefile.write("%s\t%d\n" % (k, v))


if __name__ == '__main__':
    train = pd.read_csv(TRAIN_DATA_FILE)
    test = pd.read_csv(TEST_DATA_FILE)

    toxic1 = pd.read_csv('../experiment2/train_toxicity.csv')
    toxic2 = pd.read_csv('../experiment2/attack_aggression.csv')

    get_X_train_X_test(train, test, toxic1, toxic2)


    # train = pd.read_csv(TRAIN_DATA_FILE)
    # test = pd.read_csv(TEST_DATA_FILE)
    # X_train, X_test, word_index = preprocessing.get_X_train_X_test(train, test)
    # embedding_matrix = preprocessing.get_embedding_matrix(word_index,GLOVE_EMBEDDING_FILE)
    # np.save(embedding_matrix_path, embedding_matrix)
    # y = get_Y(train)

    # print('random')
    # perm = np.random.permutation(len(y))
    # X_train=X_train[perm]
    # y=y[perm]



    # preprocessing.dump_X_Y_train_test(X_train, X_test, y, word_index)
    # X_train, X_test, y, word_index=preprocessing.load_train_test_y()
    # embedding_matrix1=np.load(embedding_matrix_path)
    # print(X_train.shape)
    # print(X_test.shape)
    # print(y.shape)
    # print(len(word_index))
    # print(embedding_matrix1.shape)
    # with open('word_index.txt','w') as fp:
    #    for word,index in word_index.items():
    #        fp.write(word+"\n")
