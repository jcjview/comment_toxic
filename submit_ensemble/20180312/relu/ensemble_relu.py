import pandas as pd

path = './'

sup = pd.read_csv('0.9883_relu_cnn_lstm_cv_craw_300_0311.csv')
p1 = pd.read_csv('0.9877_relu_cnn_lstm_cv_craw_150.csv')
p2 = pd.read_csv('0.9874_relu_cnn_lstm_cv_300_0305.csv')
p3 = pd.read_csv('../duo/0.9874_relu_cnn_lstm_cv.csv')
# p4 = pd.read_csv('0.9883_relu_cnn_lstm_cv_craw_300_0311.csv')

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
blend[col] = 0.25 * minmax_scale(sup[col].values) + \
             0.25 * minmax_scale(p1[col].values) + \
             0.25 * minmax_scale(p2[col].values) + \
             0.25 * minmax_scale(p3[col].values)
             # 0.2 * minmax_scale(p4[col].values)

print('stay tight kaggler')
blend.to_csv("ensemble_relu.csv", index=False)
