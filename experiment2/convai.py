import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import config
from sklearn.utils import shuffle
toxic_cmt = pd.read_table('../input/toxicity_annotated_comments.tsv')
toxic_annot = pd.read_table('../input/toxicity_annotations.tsv')

def JoinAndSanitize(cmt, annot):
    df = cmt.set_index('rev_id').join(annot.groupby(['rev_id']).mean())
    df = Sanitize(df)
    return df

def Sanitize(df):
    comment = 'comment' if 'comment' in df else 'comment_text'
    df[comment] = df[comment].str.lower().str.replace('newline_token', ' ')
    df[comment] = df[comment].fillna('NA')
    return df

toxic = JoinAndSanitize(toxic_cmt, toxic_annot)
toxic = shuffle(toxic)
toxic.to_csv('train_toxicity.csv',encoding='utf-8')

# train_orig = pd.read_csv(config.TRAIN_DATA_FILE)
# test_orig = pd.read_csv(config.TEST_DATA_FILE)
#
# train_orig = Sanitize(train_orig)
# test_orig = Sanitize(test_orig)