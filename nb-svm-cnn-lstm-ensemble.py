import numpy as np, pandas as pd

f_baseline = 'submission_w2v_lstm.csv'
f_lstm = 'submission-LSTM.csv'
f_nbsvm = 'submission_nb-svm.csv'

p_baseline=pd.read_csv(f_baseline)
p_lstm = pd.read_csv(f_lstm)
p_nbsvm = pd.read_csv(f_nbsvm)

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
p_res = p_lstm.copy()
p_res[label_cols] = (p_baseline[label_cols]+p_nbsvm[label_cols] + p_lstm[label_cols]) / 3

p_res.to_csv('submission-emsemble.csv', index=False)