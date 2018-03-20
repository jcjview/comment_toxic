import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
def fit_cv(X, y, n_splits=10):
    estimators, scores = [], []
    kf = KFold(n_splits=n_splits)
    for train, valid in kf.split(X):
        X_train_ = X[train]
        y_train_ = y[train]
        X_valid_ = X[valid]
        y_valid_ = y[valid]

        estimators_fold = []
        for i in tqdm(range(6)):
            y_train_one_label = y_train_[:, i]
            estimator = CatBoostClassifier(iterations=500,
                                           learning_rate=0.02,
                                           depth=2,
                                           verbose=False)
            estimator.fit(X_train_, y_train_one_label)
            estimators_fold.append(estimator)
        estimators.append(estimators_fold)

        y_valid_pred = []
        for estimator in estimators_fold:
            y_valid_pred_one_label = estimator.predict_proba(X_valid_)
            y_valid_pred.append(y_valid_pred_one_label)
        y_valid_pred = np.stack(y_valid_pred, axis=1)[..., 1]
        score = roc_auc_score(y_valid_, y_valid_pred)
        scores.append(score)
    return scores, estimators


scores, estimators = fit_cv(X_valid, y_valid_multilabel)