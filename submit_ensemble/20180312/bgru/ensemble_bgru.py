import pandas as pd

path = './'

sup = pd.read_csv('0.9887_bgru1_roll_cv_fold10_craw_300.csv')#9851
p1 = pd.read_csv('0.9885_bgru1_cv_fold10_0302glove.csv')#9853
p2 = pd.read_csv('0.9881_bgru1_cv_fold10_fit_generatorglove 150 final_craw-150 new_experiment 0312.csv')#9852

p3 = pd.read_csv('0.9888_bgru1_roll_cv_fold10.csv')#9854
# p4 = pd.read_csv('0.9881_bgru1_cv_fold10_fit_generator.csvglove 150 final_craw-150 new_experiment 0312.csv')
p4 = pd.read_csv('0.9882_bgru1_roll_cv_fold10_final_level_150_0312.csv')#9850

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
blend[col] = 0.1 * minmax_scale(sup[col].values) + \
             0.15 * minmax_scale(p1[col].values) + \
             0.2 * minmax_scale(p2[col].values) + \
             0.45 * minmax_scale(p3[col].values) + \
             0.1 * minmax_scale(p4[col].values)

print('stay tight kaggler')
blend.to_csv("ensemble_bgru_v2.zip", index=False,float_format='%.8f', compression='gzip')#0.9859
