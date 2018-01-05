
import pandas as pd
import urllib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# download annotated comments and annotations


comments = pd.read_csv('attack_annotated_comments.tsv', sep = '\t', index_col = 0)
annotations = pd.read_csv('attack_annotations.tsv',  sep = '\t')
len(annotations['rev_id'].unique())
# labels a comment as an atack if the majority of annoatators did so
labels = annotations.groupby('rev_id')['attack'].mean()

annotations = pd.read_csv('aggression_annotations.tsv',  sep = '\t')
aggression_labels = annotations.groupby('rev_id')['aggression'].mean()

annotations = pd.read_csv('toxicity_annotations.tsv',  sep = '\t')
toxicity_labels = annotations.groupby('rev_id')['toxicity'].mean()

result = pd.concat([comments, labels,aggression_labels], axis=1)

# result=result.loc[:, ['rev_id', 'comment','toxicity','attack','aggression']]
# result=result[result['toxicity']>0]
result.to_csv('output_test.csv',encoding='utf-8')