import numpy as np, pandas as pd
import scipy

path='./'
f1 =path+ 'lstm_cap_0.9841.csv'
f2 = path+'gru_cap_crawl_0.9840.csv'
f3 = path+'glove_gru_cap_0.9841.csv'

f11 = path+'9848_bgru1_cv0.9866.csv'
f12 = path+'9841_glove_dpcnn_cv0.9835.csv'
f13 = path+'9840_gru_gmp_cv0.9872.csv'

p1=pd.read_csv(f1)
p2 = pd.read_csv(f2)
p3 = pd.read_csv(f3)
# p4=pd.read_csv(f4)

p11=pd.read_csv(f11)
p12 = pd.read_csv(f12)
p13 = pd.read_csv(f13)

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
p_res = p1.copy()
p_res2 = p1.copy()
p_res3 = p1.copy()
p_res[label_cols] = (p1[label_cols]+p2[label_cols] + p3[label_cols]+
                     p11[label_cols]+p12[label_cols] + p13[label_cols]) / 6

p_res2[label_cols] = (0.9841*p1[label_cols]+
                      0.984*p2[label_cols] +
                      0.9841 *p3[label_cols]+
                      0.9848 *p11[label_cols]+
                      0.9841 *p12[label_cols] +
                      0.9840 *p13[label_cols]) / \
                     (0.9841+0.984+0.9841+0.9848+0.9841+0.9840)

p_res.to_csv('submission-emsemble-avg-20180212.csv', index=False)

p_res2.to_csv('submission-emsemble-avg-20180212_v2.csv', index=False)

p_res3[label_cols] = (p1[label_cols]*p2[label_cols] * p3[label_cols]*
                     p11[label_cols]*p12[label_cols] * p13[label_cols]) **(1/6)

p_res3.to_csv('submission-emsemble-avg-20180212_v3.csv', index=False)