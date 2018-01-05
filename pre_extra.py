
# coding: utf-8

import sys, os, re, csv, codecs, numpy as np, pandas as pd
path = './data/'

extra_TRAIN_DATA_FILE=path+"toxicity_annotated_comments.tsv"
extra_test_DATA_FILE1=path+"toxicity_annotations.tsv"
extra_TRAIN_DATA_FILE2=path+"toxicity_worker_demographics.tsv"


train = pd.read_csv(extra_TRAIN_DATA_FILE, sep='\t')

train1 = pd.read_csv(extra_test_DATA_FILE1, sep='\t')
train2 = pd.read_csv(extra_TRAIN_DATA_FILE2, sep='\t')
result = pd.concat([train, train1,train2], axis=1)

result=result.loc[:, ['rev_id', 'comment','toxicity']]


# In[15]:


result.head()


# In[25]:


result.to_csv('output_test.csv',encoding='utf-8')


# In[17]:





# In[19]:




