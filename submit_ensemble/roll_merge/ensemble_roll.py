import pandas as pd

path = './'

# sup = pd.read_csv('0.9884_pooled_gru_cnn_cv_fold10.csv')
p1 = pd.read_csv('0.9884_pooled_gru_cnn_cv_fold10.csv')
p2 = pd.read_csv('0.9887_pooled_gru_cnn_roll_cv_fold10.csv')
# p3 = pd.read_csv('0.9874_bgru1_cv_fold10_0215.csv')
# p4 = pd.read_csv('0.9845_lstm_neptune_cv_fold10.csv')
# p5 = pd.read_csv('0.9892_capsule_cv_fold10.csv')
# p5 = pd.read_csv('ensemble_capsule.csv')

# p7 = pd.read_csv('9855_glove_500_capsule.csv')
#
# p8 = pd.read_csv('9854_crawl_500_capsule.csv')

# All credits goes to original authors.. Just another blend...
from sklearn.preprocessing import minmax_scale

blend = p1.copy()
col = blend.columns

col = col.tolist()
col.remove('id')
# keeping weight of single best model higher than other blends..
blend[col] = 0.5 * minmax_scale(p1[col].values) + \
             0.5* minmax_scale(p2[col].values)

print('stay tight kaggler')
blend.to_csv("ensemble_gru_cnn_roll_unroll.csv", index=False)
