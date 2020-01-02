# import pandas as pd
from sklearn.model_selection import train_test_split

from modeling_pipeline import pipeline
from modeling_pipeline.processing.data_manager import data_loader, save_pipeline
from modeling_pipeline.config import config
from modeling_pipeline import __version__ as _version

import logging

_logger = logging.getLogger(__name__)


def run_training():
    data = data_loader(file_name=config.TRAINING_DATA_FILE)

    trainx, validx, trainy, validy = train_test_split(
        data.drop(config.TARGET, axis=1),
        data[config.TARGET],
        test_size=0.1,
        random_state=0)

    trainx, trainy = pipeline.logtransformer(trainx, trainy)
    validx, validy = pipeline.logtransformer(validx, validy)

    pipeline.pipeline.fit(trainx, trainy)

    _logger.info(f'saving model version: {_version}')
    save_pipeline(pipeline_to_persist=pipeline.pipeline)


if __name__ == '__main__':
    run_training()
    print('completed')
