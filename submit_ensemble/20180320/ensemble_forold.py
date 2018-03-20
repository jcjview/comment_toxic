import pandas as pd

path = './'

p1 = pd.read_csv('submit_ensemble/20180320/ensemble_bgru_0320v1.csv')
p2 = pd.read_csv('submit_ensemble/20180320/ensemble_pool_gru_0320v1.csv')
p3 = pd.read_csv('submit_ensemble/20180319/ensemble_capsule0319_9865.csv')
p4 = pd.read_csv('submit_ensemble/20180319/blend_it_all9868.csv')

# All credits goes to original authors.. Just another blend...
from sklearn.preprocessing import minmax_scale

blend = p1.copy()
col = blend.columns

col = col.tolist()
col.remove('id')
# keeping weight of single best model higher than other blends..
blend[col] = 0.2 * minmax_scale(p1[col].values) + \
             0.3 * minmax_scale(p2[col].values) + \
             0.3 * minmax_scale(p3[col].values) + \
             0.2 * minmax_scale(p4[col].values)

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
blend.to_csv("submit_ensemble/20180320/ensemble_0320finalv6_ms.csv.gz", index=False, float_format='%.9f',
             compression='gzip')
