import pandas as pd

path = './'

sup = pd.read_csv('submit_ensemble/20180306/hight_of_blend_v2.csv')
# p1 = pd.read_csv('0.9882_bgru1_roll_cv_fold10_final_craw_150.csv')
# p2 = pd.read_csv('0.9888_bgru1_roll_cv_fold10.csv')
# p3 = pd.read_csv('0.9881_pooled_gru_cnn_cv_fold10.csv')
# p4 = pd.read_csv('0.9845_lstm_neptune_cv_fold10.csv')
# p5 = pd.read_csv('0.9892_capsule_cv_fold10.csv')

p6 = pd.read_csv('submit_ensemble/20180312/9859_glove_300_capsule.csv')

p7 = pd.read_csv('submit_ensemble/20180306/9858_glove_500_capsule.csv')

p8 = pd.read_csv('submit_ensemble/20180306/9855_glove_500_capsule.csv')

# All credits goes to original authors.. Just another blend...
from sklearn.preprocessing import minmax_scale

blend = sup.copy()
col = blend.columns

col = col.tolist()
col.remove('id')
# keeping weight of single best model higher than other blends..
blend[col] =0.5*minmax_scale(p6[col].values) + \
            0.25*minmax_scale(p7[col].values) + \
            0.25 *minmax_scale(p8[col].values)


print('stay tight kaggler')
blend.to_csv("submit_ensemble/20180312/ensemble_capsule0312.csv", index=False)
