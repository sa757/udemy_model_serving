# -*- coding: utf-8 -*-
from modeling_pipeline.config import config

import pandas as pd


def validate_inputs(input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for unprocessable values."""

    validated_data = input_data.copy()

    # check for numerical variables with NA not seen during training
    if input_data[config.NUMERICAL_VARS_WITH_NA].isnull().any().any():
        validated_data = validated_data.dropna(axis=0, subset=config.NUMERICAL_VARS_WITH_NA)

    # check for categorical variables with NA not seen during training
    if input_data[config.CATEGORICAL_VARS_WITH_NA].isnull().any().any():
        validated_data = validated_data.dropna(axis=0, subset=config.CATEGORICAL_VARS_WITH_NA)

    # check for values <= 0 for the log transformed variables
    if (input_data[config.LOG_VARS] <= 0).any().any():
        vars_with_neg_values = config.LOG_VARS[(input_data[config.LOG_VARS] <= 0).any()]
        validated_data = validated_data[validated_data[vars_with_neg_values] > 0]
    return validated_data
