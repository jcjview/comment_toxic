import pandas as pd

path = './'

sup = pd.read_csv('submit_ensemble/20180319/blend_it_all9868.csv')#9867
p1 = pd.read_csv('submit_ensemble/20180319/ensemble_bgru_0319v1_9863.csv')#9863
p2 = pd.read_csv('submit_ensemble/20180312/duo/ensemble_duo_0316.csv')#0.9856
p3 = pd.read_csv('submit_ensemble/20180312/pool/ensemble_pool_gru_0315v4.csv')#~9863
p4 = pd.read_csv('submit_ensemble/20180312/duo/0.9872_lstm_neptune_cv_fold10.csv')#9856
p5 = pd.read_csv('submit_ensemble/20180319/ensemble_capsule0319_9865.csv')#9865
p6 = pd.read_csv('submit_ensemble/20180312/pool/0.9899_pooled_gru_cnn1_cv_fold10_preprocess_glove_300_0316_9860.csv')#~9860
p7 = pd.read_csv('submit_ensemble/20180312/two_rnn/0.9891_two_rnn_cnn_2e_cv_fold10_glove_300_prep_9858.csv')
#
p8 = pd.read_csv('submit_ensemble/20180319/yuduo-0.9898_pooled_gru_cnn1_cv_fold10_glove_400_9861.csv')
p9 = pd.read_csv('submit_ensemble/20180319/0.9895_bgru1_cv_fold10_dr4_0319.csv')
p10 = pd.read_csv('submit_ensemble/20180319/0.9896_pooled_gru_cnn1_cv_fold10_craw_300_0319_9856.csv')

# All credits goes to original authors.. Just another blend...
from sklearn.preprocessing import minmax_scale

blend = sup.copy()
col = blend.columns

col = col.tolist()
col.remove('id')
# keeping weight of single best model higher than other blends..
blend[col] = \
    (0.2 *minmax_scale (sup[col].values) + \
    0.1 *minmax_scale (p1[col].values) + \
    0.05 *minmax_scale (p2[col].values) + \
    0.2 * minmax_scale(p3[col].values) + \
    0.05 *minmax_scale (p4[col].values) + \
    0.2 * minmax_scale(p5[col].values) + \
    0.1 * minmax_scale(p6[col].values) +\
    0.1 * minmax_scale(p7[col].values) +\
    0.1 * minmax_scale(p8[col].values) + \
    0.05 * minmax_scale(p9[col].values) + \
    0.05 * minmax_scale(p10[col].values)) /(0.2+0.1+0.05+0.2+0.05+0.2+0.1+0.1+0.1+0.05+0.05)

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
blend.to_csv("submit_ensemble/20180319/ensemble_0320finalv2_ms.csv.gz", index=False,float_format='%.9f', compression='gzip')
