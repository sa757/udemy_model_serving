from sklearn.externals import joblib

from img_model_package import pipeline as pipe
from img_model_package.modeling_pipeline.config import config
from img_model_package.modeling_pipeline.processing import data_management as dm
from img_model_package.modeling_pipeline.processing import preprocessors as pp


def run_training(save_result: bool = True):
    """Train a Convolutional Neural Network."""

    images_df = dm.load_image_paths(config.DATA_FOLDER)
    X_train, X_test, y_train, y_test = dm.get_train_test_target(images_df)

    enc = pp.LabelEncoder()
    enc.fit(y_train)
    y_train = enc.transform(y_train)

    pipe.model_pipeline.fit(X_train, y_train)

    if save_result:
        joblib.dump(enc, config.ENCODER_PATH)
        dm.save_pipeline_keras(pipe.model_pipeline)


if __name__ == '__main__':
    run_training(save_result=True)
