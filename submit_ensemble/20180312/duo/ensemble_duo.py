import pandas as pd

path = './'

sup = pd.read_csv('0.9872_lstm_neptune_cv_fold10.csv')
p1 = pd.read_csv('0.9866_lstm_neptune_cv_fold10.csv')
p2 = pd.read_csv('0.9861_lstm_neptune_cv_fold10_len200.csv')
# p3 = pd.read_csv('0.9881_pooled_gru_cnn_cv_fold10.csv')
# p4 = pd.read_csv('0.9845_lstm_neptune_cv_fold10.csv')
# p5 = pd.read_csv('0.9892_capsule_cv_fold10.csv')
# 0.9889_pooled_gru_cnn_cv_fold10.csv
# 0.9888_capsule_cv_fold10.csv
# 0.9886_bgru1_roll_cv_fold10.csv
# 0.9885_bgru1_roll_cv_fold10_len200.csv
# 0.9879_bgru1_roll_cv_fold10.csv
# 0.9875_pooled_gru_cnn_cv_fold10.csv
# 0.9874_relu_cnn_lstm_cv.csv
# 0.9872_lstm_neptune_cv_fold10.csv
# 0.9870_ATTENTION_lstm_dropout1_cv.csv
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
blend[col] =0.5*minmax_scale(sup[col].values) + \
            0.25*minmax_scale(p1[col].values) + \
            0.25 *minmax_scale(p2[col].values)


print('stay tight kaggler')
blend.to_csv("ensemble_duo_lstm_neptune.csv", index=False)
