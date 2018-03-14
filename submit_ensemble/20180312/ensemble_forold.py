import pandas as pd

path = './'

sup = pd.read_csv('submit_ensemble/20180312/blend_it_all.csv')#9867
p1 = pd.read_csv('submit_ensemble/20180312/bgru/ensemble_bgru_v1.csv')#9859
# p2 = pd.read_csv('submit_ensemble/20180312/relu/0.9883_relu_cnn_lstm_cv_craw_300_0311.csv')#0.9848
p3 = pd.read_csv('submit_ensemble/20180312/pool/ensemble_pool_gru_v1.csv')#9861
p4 = pd.read_csv('submit_ensemble/20180312/duo/0.9872_lstm_neptune_cv_fold10.csv')#9855
# p5 = pd.read_csv('0.9892_capsule_cv_fold10.csv')
# p5 = pd.read_csv('ensemble_capsule.csv')
p5 = pd.read_csv('submit_ensemble/20180312/ensemble_capsule0312.csv')#9859
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
blend[col] = \
    0.35 * minmax_scale(sup[col].values) + \
    0.15 *minmax_scale (p1[col].values) + \
    0.25 *minmax_scale (p3[col].values) + \
    0.1 * minmax_scale(p4[col].values) + \
    0.15 * minmax_scale(p5[col].values)

    # 0.1 * (p2[col].values) + \

# blend[col] = \
#     0.25 * minmax_scale(sup[col].values) + \
#     0.1 * minmax_scale(p1[col].values) + \
#     0.1 * minmax_scale(p2[col].values) + \
#     0.15 * minmax_scale(p3[col].values) + \
#     0.15 * minmax_scale(p4[col].values) + \
#     0.25 * minmax_scale(p5[col].values)
# 0.25 * minmax_scale(p6[col].values) + \
# 0.1 * minmax_scale(p7[col].values) + \
# 0.1 * minmax_scale(p8[col].values)

# / (0.3 + 0.1 + 0.2 + 0.2 + 0.1 + 0.1 + 0.25 + 0.1 + 0.1)

print('stay tight kaggler')
blend.to_csv("ensemble_0314v2_ms.csv", index=False,float_format='%.8f')
