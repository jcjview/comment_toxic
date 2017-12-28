import numpy as np, pandas as pd

f_lstm = 'baseline.csv'
f_nbsvm = 'submission_nb-svm.csv'

p_lstm = pd.read_csv(f_lstm)
p_nbsvm = pd.read_csv(f_nbsvm)

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
p_res = p_lstm.copy()
p_res[label_cols] = (p_nbsvm[label_cols] + p_lstm[label_cols]) / 2

p_res.to_csv('submission-emsemble.csv', index=False)