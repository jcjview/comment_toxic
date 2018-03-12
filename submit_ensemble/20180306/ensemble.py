import pandas as pd

path = './'

sup = pd.read_csv('hight_of_blend_v2.csv')
p1 = pd.read_csv('ensemble_bgru.csv')
p2 = pd.read_csv('0.9874_relu_cnn_lstm_cv.csv')
p3 = pd.read_csv('0.9881_pooled_gru_cnn_cv_fold10.csv')
p4 = pd.read_csv('0.9845_lstm_neptune_cv_fold10.csv')
# p5 = pd.read_csv('0.9892_capsule_cv_fold10.csv')
p5 = pd.read_csv('ensemble_capsule.csv')

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
blend[col] = (0.25 * minmax_scale(sup[col].values) + \
              0.2 * minmax_scale(p1[col].values) + \
              0.1 * minmax_scale(p2[col].values) + \
              0.1 * minmax_scale(p3[col].values) + \
              0.1 * minmax_scale(p4[col].values) + \
              0.25 * minmax_scale(p5[col].values)
              # 0.25 * minmax_scale(p6[col].values) + \
              # 0.1 * minmax_scale(p7[col].values) + \
              # 0.1 * minmax_scale(p8[col].values)
              )
             # / (0.3 + 0.1 + 0.2 + 0.2 + 0.1 + 0.1 + 0.25 + 0.1 + 0.1)

print('stay tight kaggler')
blend.to_csv("ensemble_0306v2.csv", index=False)
