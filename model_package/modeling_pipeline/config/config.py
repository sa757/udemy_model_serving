# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 10:07:07 2019

@author: kim85
"""

import os
import pathlib
import zipfile
import modeling_pipeline

import pandas as pd

pd.options.display.max_rows = 10
pd.options.display.max_columns = 10

PACKAGE_ROOT = pathlib.Path(modeling_pipeline.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'
DATASET_DIR = PACKAGE_ROOT / 'datasets'
ZIPFILE_DIR = DATASET_DIR / 'kaggle_house_price.zip'

with zipfile.ZipFile(ZIPFILE_DIR, 'r') as f:
    f.extractall(DATASET_DIR)

TRAINING_DATA_FILE = 'train.csv'
TESTING_DATA_FILE = 'test.csv'
TARGET = 'SalePrice'

PIPELINE_NAME = 'lasso_model'
PIPELINE_SAVE_FILE = f'{PIPELINE_NAME}_pipeline_v'

FEATURES = [
        'MSSubClass', 'MSZoning', 'Neighborhood', 'OverallQual', 
        'OverallCond', 'YearRemodAdd', 'RoofStyle', 'MasVnrType', 
        'BsmtQual', 'BsmtExposure', 'HeatingQC', 'CentralAir', 
        '1stFlrSF', 'GrLivArea', 'BsmtFullBath', 'KitchenQual', 
        'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageFinish', 
        'GarageCars', 'PavedDrive', 'LotFrontage'
        ]

CATEGORICAL_VARS_WITH_NA = ['MasVnrType', 'BsmtQual', 'BsmtExposure', 
                            'FireplaceQu', 'GarageType', 'GarageFinish']
NUMERICAL_VARS_WITH_NA = ['LotFrontage']

CATEGOCIAL_VARS = [
        'MSZoning', 'Neighborhood', 'RoofStyle', 'MasVnrType', 'BsmtQual', 
        'BsmtExposure', 'HeatingQC', 'CentralAir', 'KitchenQual', 
        'FireplaceQu', 'GarageType', 'GarageFinish', 'PavedDrive'
        ]

TEMPORAL_VARS = ['YearRemodAdd']
REFERENCE_VAR = 'YrSold'

DROP_VARS =[
        'Foundation', 'HalfBath', 'Id', '3SsnPorch', 'KitchenAbvGr', 
        'BsmtFinSF1', 'HouseStyle', 'SaleCondition', 'ExterQual', 
        'TotRmsAbvGrd', 'Utilities', 'Condition2', 'Street', 'Electrical', 
        'Exterior2nd', 'Fence', 'BldgType', 'RoofMatl', 'MoSold', 'BsmtCond',
        'BsmtUnfSF', 'SaleType', 'BedroomAbvGr', 'MiscFeature', 'Functional', 
        'LandSlope', 'Heating', 'BsmtFinSF2', 'LotShape', 'Exterior1st', 
        'GarageQual', 'OpenPorchSF', 'ScreenPorch', 'PoolQC', 'YrSold', 
        'LandContour', 'YearBuilt', 'ExterCond', 'LotArea', 'GarageCond', 
        'GarageArea', 'LowQualFinSF', 'BsmtFinType2', 'BsmtHalfBath', 
        'TotalBsmtSF', 'GarageYrBlt', 'FullBath', 'PoolArea', 'LotConfig', 
        '2ndFlrSF', 'BsmtFinType1', 'Condition1', 'WoodDeckSF', 'MasVnrArea', 
        'Alley', 'MiscVal', 'EnclosedPorch']

LOG_VARS = ['LotFrontage', '1stFlrSF', 'GrLivArea', 'SalePrice']




