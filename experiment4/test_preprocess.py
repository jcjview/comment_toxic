from config import *
import util.preprocessing as preprocessing
import pandas as pd
import numpy as np
def get_Y(train):
    return train[CLASSES_LIST].values
embedding_matrix_path='temp.npy'
if __name__ == '__main__':
    train = pd.read_csv(TRAIN_DATA_FILE)
    test = pd.read_csv(TEST_DATA_FILE)
    X_train, X_test, word_index = preprocessing.get_X_train_X_test(train, test)
    embedding_matrix = preprocessing.get_embedding_matrix(word_index,GLOVE_EMBEDDING_FILE)
    np.save(embedding_matrix_path, embedding_matrix)
    
    y = get_Y(train)
    preprocessing.dump_X_Y_train_test(X_train, X_test, y, word_index)
    preprocessing.load_train_test_y()
    embedding_matrix1=np.load(embedding_matrix_path)