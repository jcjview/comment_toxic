import pandas as pd

path = './'

sup = pd.read_csv('0.9880_gru_gmp_gav_cv1_craw_150.csv')
p1 = pd.read_csv('0.9876_ATTENTION_lstm_dropout1_cv_craw_300_0311.csv')
p2 = pd.read_csv('0.9873_ATTENTION_lstm_dropout1_cv_craw-150.csv')
p3 = pd.read_csv('../google/0.9878_bgru1_roll_cv_fold10_google_300_0308.csv')
p4 = pd.read_csv('../google/0.9873_pooled_gru_cnn_cv_fold10_google_300.csv')
p5 = pd.read_csv('../google/0.9860_ATTENTION_lstm_dropout1_cv_google_300_0311.csv')
p6 = pd.read_csv('../duo/0.9889_pooled_gru_cnn_cv_fold10.csv')
p7 = pd.read_csv('../duo/0.9870_ATTENTION_lstm_dropout1_cv.csv')
#


#




p8 = pd.read_csv('../duo/0.9861_ATTENTION_lstm_dropout1_cv.csv')

p9 = pd.read_csv('../duo/0.9875_pooled_gru_cnn_cv_fold10.csv')

# p8 = pd.read_csv('9854_crawl_500_capsule.csv')

# All credits goes to original authors.. Just another blend...
from sklearn.preprocessing import minmax_scale

blend = sup.copy()
col = blend.columns

col = col.tolist()
col.remove('id')
# keeping weight of single best model higher than other blends..
blend[col] = 0.1 * minmax_scale(sup[col].values) + \
             0.1 * minmax_scale(p1[col].values) + \
             0.1 * minmax_scale(p2[col].values) + \
             0.1 * minmax_scale(p3[col].values) + \
             0.1 * minmax_scale(p4[col].values) + \
             0.1 * minmax_scale(p5[col].values) + \
             0.1 * minmax_scale(p6[col].values) + \
             0.1 * minmax_scale(p7[col].values) + \
             0.1 * minmax_scale(p8[col].values)+\
             0.1 * minmax_scale(p9[col].values)
             # 0.2 * minmax_scale(p4[col].values)

print('stay tight kaggler')
blend.to_csv("ensemble_other.csv", index=False)
