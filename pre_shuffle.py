
import config

import pandas as pd
from sklearn.utils import shuffle
train =pd.DataFrame(pd.read_csv(config.TRAIN_DATA_FILE))
train["comment_text"].fillna("unknown", inplace=True)
train = shuffle(train)
valid=train.sort_values(["toxic"],ascending=False,kind='mergesort')
valid.to_csv('../data/train_valid.csv',encoding='utf-8')