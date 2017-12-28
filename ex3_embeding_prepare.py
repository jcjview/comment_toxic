#encoding=utf-8

import re
import sys
import codecs
import os.path
import glob
import pandas as pd
import numpy as np
import h5py
import pickle
i = 1
# word_dict = {}
# with open('word_dict.pkl3', 'rb') as fpkl:
#     word_dict=pickle.load(fpkl)
from config import  *


weights = np.zeros([max_features,embedding_dims],dtype=np.float32)
f=open('w2v.txt','r',encoding='utf-8')
f.readline()
for line in f:
    s = line.split(' ')[1:]
    for j in range(embedding_dims):
        weights[i,j] = float(s[j])
    if i%5000 == 0:
        sys.stdout.flush()
        sys.stdout.write("#")
    i=i+1
f.close()
print (i-1)

print ("OK!")

np.savez("w2v_embedding_layer.npz",weights)
print ("save word2vec weights OK")
