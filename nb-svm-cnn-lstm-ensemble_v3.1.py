import numpy as np, pandas as pd
path='./submit_achive/'
f1 =path+ '0.0433_blstm_finetuning_0.043.csv'
f2 = path+'submission-glove300.csv'
f3 = path+'0.037_cnn_finetuning_0.046.csv'
f4 = path+'0.0435_simple_lstm_glove_vectors_0.25_0.25_0.044.csv'

p1=pd.read_csv(f1)
p2 = pd.read_csv(f2)
p3 = pd.read_csv(f3)
p4=pd.read_csv(f4)
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
p_res = p1.copy()

p_res[label_cols] = (p1[label_cols]+p2[label_cols] + p3[label_cols]+p4[label_cols]) / 4

p_res.to_csv('submission-emsemble-finetuning_v3.1.csv', index=False)