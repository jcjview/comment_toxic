#encoding=utf-8

import logging
import sys
import multiprocessing
import numpy as np

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models import doc2vec

from config import *

if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


r = np.random.randint(100000,999999,size = (1,))
print (r[0])
sents = doc2vec.TaggedLineDocument("./all.txt")
print (sents)
model = doc2vec.Doc2Vec(sents, size = embedding_dims, window = 9, min_count=5, iter=45, hs=0, negative=11, seed=r[0],)
model.wv.save_word2vec_format("w2v.txt", binary=False)
model.save("d2v.model")

