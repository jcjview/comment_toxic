import numpy as np
import pandas as pd
import config
from tensorflow.python.training import moving_averages
sample_submission = pd.read_csv("0.0391_5_model_bagging2.0_ex.csv")
test_predicts=sample_submission[config.CLASSES_LIST].values
PROBABILITIES_NORMALIZE_COEFFICIENT = 1.3
sample_submission[config.CLASSES_LIST] **= PROBABILITIES_NORMALIZE_COEFFICIENT
# test_predicts=np.log(test_predicts)
# test_predicts-=0.5
# test_predicts=np.exp(test_predicts)
sample_submission[config.CLASSES_LIST] = test_predicts
sample_submission.to_csv('postprocessing_13_0.0391_5_model_bagging2.0_ex.csv', index=False)

