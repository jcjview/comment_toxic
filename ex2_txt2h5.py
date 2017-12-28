import h5py
import numpy as np
import pandas as pd
import pickle
CLASSES_LIST = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
MAX_TEXT_LENGTH = 100
word_dict={}
with open('./w2v.txt', 'r',encoding='utf-8') as f:
    i = 1
    line = f.readline()
    print(line.strip())
    for line in f:
        s = line.split(' ')
        ss = s[0].strip()
        word_dict[ss] = i
        i=i+1

with open('word_dict.pkl3', 'wb') as f:
    pickle.dump(word_dict, f)

def get_Y(train):
    return train[CLASSES_LIST].values

train_rows = 95851
test_rows = 226998

train = pd.read_csv("./data/train.csv")
outh5file = h5py.File('./data/train_label.h5', 'w')
train_label = outh5file.create_dataset(name='train_label',shape=(train_rows,len(CLASSES_LIST)), dtype=np.int8)
train_label[:] = get_Y(train)[:, :]
outh5file.flush()
outh5file.close()
print("OK")


outh5file = h5py.File('./data/train_token.h5','w')
train_token = outh5file.create_dataset('train_token', (train_rows,MAX_TEXT_LENGTH),dtype=np.int32)
row=0
with open('./train_corps.txt', 'r',encoding='utf-8') as f:
    for line in f:
        s = line.split()
        vec = np.zeros(MAX_TEXT_LENGTH, np.int32)
        j = 0
        for x in s:
            if j >= MAX_TEXT_LENGTH:
                break
            if x in word_dict:
                vec[j] = word_dict[x]
                j = j + 1
        train_token[row, :] = vec
        row+=1
outh5file.flush()
outh5file.close()

outh5file = h5py.File('./data/test_token.h5','w')
train_token = outh5file.create_dataset('test_token', (test_rows,MAX_TEXT_LENGTH),dtype=np.int32)
row=0
with open('./test_corps.txt', 'r',encoding='utf-8') as f:
    for line in f:
        s = line.split()
        vec = np.zeros(MAX_TEXT_LENGTH, np.int32)
        j = 0
        for x in s:
            if j >= MAX_TEXT_LENGTH:
                break
            if x in word_dict:
                vec[j] = word_dict[x]
                j = j + 1
        train_token[row, :] = vec
        row+=1
outh5file.flush()
outh5file.close()