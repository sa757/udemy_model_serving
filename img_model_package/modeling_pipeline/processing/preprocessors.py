import numpy as np
import cv2
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin


class LabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, encoder=LabelBinarizer()):
        self.encoder = encoder

    def fit(self, X, y=None):
        # 여기서는 X = 레이블임.
        self.encoder.fit(X)
        return self

    def transform(self, X):
        X = X.copy()
        X = self.encoder.transform(X)
        return X


class CreateDataset(BaseEstimator, TransformerMixin):
    def __init__(self, IMG_SIZE):
        self.img_size = IMG_SIZE

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        data_home = np.zeros((len(X), self.img_size, self.img_size, 3))
        for n in range(len(X)):
            im = cv2.imread(X[n])
            im = cv2.resize(im, (self.img_size, self.img_size))
            data_home[n] = im
        print('Dataset Images shape: {} size: {:,}'.format(
            data_home.shape, data_home.size))
        return data_home
