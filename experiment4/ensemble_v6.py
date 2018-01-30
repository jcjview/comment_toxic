import numpy as np, pandas as pd
path='./'
f1 =path+ 'submit.csv'
f2 = path+'result.csv'
f3 = path+'keras_5model_bagging2.csv'
f4 = path+'0.0443_5_model_bagging1.0.csv'

p1=pd.read_csv(f1)
p2 = pd.read_csv(f2)
p3 = pd.read_csv(f3)
p4=pd.read_csv(f4)

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
p_res = p1.copy()
p_res[label_cols] = (6*p1[label_cols]+4*p2[label_cols] +6*p3[label_cols]+6*p4[label_cols]) / (16+6)

PROBABILITIES_NORMALIZE_COEFFICIENT = 1.4
p_res[label_cols] **= PROBABILITIES_NORMALIZE_COEFFICIENT

p_res.to_csv('submission-emsemble-postprocessing_5model_craw_capsule_v2.csv', index=False)