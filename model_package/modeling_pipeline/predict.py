# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 14:12:22 2019

@author: kim85
"""

import numpy as np
import pandas as pd

from modeling_pipeline.processing.data_manager import load_pipeline
from modeling_pipeline.processing.validation import validate_inputs
from modeling_pipeline.config import config
from modeling_pipeline import __version__ as _version

import logging
import typing as t

_logger = logging.getLogger(__name__)

pipeline_file_name = f'{config.PIPELINE_SAVE_FILE}{_version}.pkl'
_price_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, input_data: t.Union[pd.DataFrame, dict], ) -> dict:
    data = input_data
    validated_data = validate_inputs(input_data=data)

    prediction = _price_pipe.predict(validated_data)

    output = np.exp(prediction)

    results = {'predictions': output, 'version': _version}

    _logger.info(
        f'Making predictions with model version: {_version} '
        f'Inputs: {validated_data} '
        f'Predictions: {results}')

    return results
