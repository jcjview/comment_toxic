import pandas as pd

path = './'

sup = pd.read_csv('0.9891_pooled_gru_cnn_cv_fold10_500.csv')
p1 = pd.read_csv('0.9889_pooled_gru_cnn_cv_fold10.csv')
p2 = pd.read_csv('0.9888_capsule_cv_fold10.csv')
p3 = pd.read_csv('0.9887_bgru1_roll_cv_fold10_500.csv')
p4 = pd.read_csv('0.9886_capsule_cv_fold10.csv')
p5 = pd.read_csv('0.9886_bgru1_roll_cv_fold10.csv')
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
blend[col] =(minmax_scale(sup[col].values) + \
            minmax_scale(p1[col].values) + \
            minmax_scale(p2[col].values) + \
            minmax_scale(p3[col].values) + \
            minmax_scale(p4[col].values) + \
            minmax_scale(p5[col].values))/6


print('stay tight kaggler')
blend.to_csv("ensemble_duo_0316.csv.gz", index=False,float_format='%.8f', compression='gzip')#9855 not good
