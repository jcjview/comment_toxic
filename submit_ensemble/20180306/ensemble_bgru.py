import pandas as pd

path = './'

sup = pd.read_csv('hight_of_blend_v2.csv')
p1 = pd.read_csv('0.9882_bgru1_roll_cv_fold10_final_craw_150.csv')#9850
p2 = pd.read_csv('0.9888_bgru1_roll_cv_fold10.csv')#9854
p3 = pd.read_csv('0.9874_bgru1_cv_fold10_0215.csv')
# p4 = pd.read_csv('0.9845_lstm_neptune_cv_fold10.csv')
# p5 = pd.read_csv('0.9892_capsule_cv_fold10.csv')
# p5 = pd.read_csv('ensemble_capsule.csv')

# p7 = pd.read_csv('9855_glove_500_capsule.csv')
#
# p8 = pd.read_csv('9854_crawl_500_capsule.csv')

# All credits goes to original authors.. Just another blend...
from sklearn.preprocessing import minmax_scale

blend = sup.copy()
col = blend.columns

col = col.tolist()
col.remove('id')
# keeping weight of single best model higher than other blends..
blend[col] = 0.35 * minmax_scale(p1[col].values) + \
             0.25 * minmax_scale(p2[col].values) + \
             0.4 * minmax_scale(p3[col].values)

print('stay tight kaggler')
blend.to_csv("ensemble_bgru.csv", index=False)
