# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 10:02:55 2019

@author: kim85
"""
import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline

from modeling_pipeline.processing import pipe_elements as pe
from modeling_pipeline.config import config
import logging

_logger = logging.getLogger(__name__)

pipeline = Pipeline([
    ('categorical_imputer',pe.CategoricalImputer(
            features=config.CATEGORICAL_VARS_WITH_NA
            )),
    ('numerical_inputer',pe.NumericalImputer(
            features=config.NUMERICAL_VARS_WITH_NA
            )),
    ('temporal_feature_estimator',pe.TemporalFeatureEstimator(
            features=config.TEMPORAL_VARS, 
            reference_feature=config.REFERENCE_VAR
            )),
    ('rare_lable_categorical_encoder',pe.RareLabelCategoricalEncoder(
            features=config.CATEGOCIAL_VARS
            )),
    ('categorical_encoder',pe.CategoricalEncoder(
            features=config.CATEGOCIAL_VARS
            )),
    ('drop_unnecessary_features',pe.DropUnnecessaryFeatures(
            features=config.DROP_VARS
            )),
    ('min_max_scaler',pe.CustomMinMaxScaler()),
    ('lasso_model',Lasso(alpha=0.005, random_state=0))
])


def logtransformer(X: pd.DataFrame, y: pd.Series, features=config.LOG_VARS):
    X, y = X.copy(), y.copy()
    for feature in features:
        if feature == y.name:
            y = np.log(y)
        else:
            X[feature] = np.log(X[feature])
    return X, y