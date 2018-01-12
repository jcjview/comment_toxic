import numpy as np, pandas as pd
path='./submit_achive/'
f1 =path+ '0.0435_simple_lstm_glove_vectors_0.25_0.25.csv'
f2 = path+'cnn_rnn_embeding.csv'
f3 = path+'submission-glove300.csv'
f4 = path+'submission_nb-svm.csv'

p1=pd.read_csv(f1)
p2 = pd.read_csv(f2)
p3 = pd.read_csv(f3)
p4=pd.read_csv(f4)
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
p_res = p1.copy()

p_res[label_cols] = (4*p1[label_cols]+3*p2[label_cols] + 2*p3[label_cols]+p4[label_cols]) / 10

p_res.to_csv('submission-emsemble-attention-cnn-lstm2.csv', index=False)