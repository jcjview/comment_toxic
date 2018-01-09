from sklearn.utils import shuffle

import config
import pandas as pd
import numpy as np


def shuffle_text(d):
    shuffle(d)
    # print(" ".join(d))
    return d


def dropout(d, p=0.5):
    len_ = len(d)
    index = np.random.choice(d, int(len_ * p))
    # print("".join(index))
    return index


# text="I'd be moving ``Mutual Assured Destruction`` to ``talk`` for not appealing to a Reagan voter's biases about its effectiveness, and for dropping the ``ly``.NEWLINE_TOKENNEWLINE"
# d = text.split()
# d = shuffle_text(d)
# d = dropout(d, p=0.5)

train = pd.DataFrame(pd.read_csv(config.TRAIN_DATA_FILE))
train["comment_text"].fillna("unknown", inplace=True)
train = shuffle(train)
valid = train.sort_values(["toxic"], ascending=False, kind='mergesort')
shuffle_data = train[train["toxic"] == 1]
for index, row in shuffle_data.iterrows():
    text = row['comment_text']
    d = text.split()
    d = shuffle_text(d)
    d = dropout(d, p=0.8)
    shuffle_data.loc[index, 'comment_text'] = " ".join(d)

shuffle_data.to_csv(config.path+'shuffle_data.csv', encoding='utf-8')
result = pd.concat([valid,shuffle_data])
result.to_csv(config.path+'train_valid_test.csv', encoding='utf-8')
