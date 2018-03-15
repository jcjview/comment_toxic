import pandas as pd

path = './'

sup = pd.read_csv('0.9885_pooled_gru_cnn_cv_fold10_trans_craw_150_0314_9853.csv')#9853

p1 = pd.read_csv('0.9886_pooled_gru_cnn_cv_fold10_craw_300.csv')#9855
p2 = pd.read_csv('0.9884_pooled_gru_cnn_cv_fold10_0308_craw_150.csv')#9855
p3 = pd.read_csv('0.9891_pooled_gru_cnn_cv_fold10_800_9856.csv')#9856
p4 = pd.read_csv('0.9885_two_rnn_cnn_cv_fold10.csv')#9855

#
#


# p6 = pd.read_csv('9858_glove_500_capsule.csv')

# p7 = pd.read_csv('9855_glove_500_capsule.csv')

# p8 = pd.read_csv('9854_crawl_500_capsule.csv')

# All credits goes to original authors.. Just another blend...
from sklearn.preprocessing import minmax_scale

blend = sup.copy()
col = blend.columns

col = col.tolist()
col.remove('id')
# keeping weight of single best model higher than other blends..
blend[col] =0.15* minmax_scale(p1[col].values) + \
            0.15 *minmax_scale(p2[col].values) + \
            0.35 *minmax_scale(p3[col].values) + \
            0.35 *minmax_scale(p4[col].values)

print('stay tight kaggler')
blend.to_csv("ensemble_pool_gru_0315v3.csv.gz", index=False,float_format='%.8f', compression='gzip')#0.9863
