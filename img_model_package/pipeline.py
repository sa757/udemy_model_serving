from sklearn.pipeline import Pipeline

from img_model_package.modeling_pipeline.config import config
from img_model_package.modeling_pipeline.processing import preprocessors as pp
from img_model_package import model

model_pipeline = Pipeline([
    ('datasets', pp.CreateDataset(config.IMAGE_SIZE)),
    ('model', model.cnn_clf)
])