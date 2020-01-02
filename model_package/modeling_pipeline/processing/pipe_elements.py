import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class CategoricalImputer(BaseEstimator, TransformerMixin):
    """Replace Categorical Column's Missing Values to text:Missing"""

    def __init__(self, features=None) -> None:
        if not isinstance(features, list):
            self.features = [features]
        else:
            self.features = features

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'fit':
        return self

    def transform(self, X: pd.DataFrame) -> 'transform':
        X = X.copy()
        for feature in self.features:
            X[feature] = X[feature].fillna('Missing')
        return X


class NumericalImputer(BaseEstimator, TransformerMixin):
    """Replace Numerical Column's Missing Values to Median"""

    def __init__(self, features=None) -> None:
        if not isinstance(features, list):
            self.features = [features]
        else:
            self.features = features

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'fit':
        self.imputer_dict_ = {feature: X[feature].mode()[0] for feature in self.features}
        # np.save('./dataset/mean_var_dict.npy', self.imputer_dict_) # 백업 반영시 
        return self

    def transform(self, X: pd.DataFrame) -> 'transform':
        X = X.copy()
        for feature in self.features:
            X[feature] = X[feature].fillna(self.imputer_dict_[feature])
        return X


class TemporalFeatureEstimator(BaseEstimator, TransformerMixin):
    """Make variable by calculating Elapsed_year of variables"""

    def __init__(self, features=None, reference_feature=None) -> None:
        if not isinstance(features, list):
            self.features = [features]
        else:
            self.features = features
        self.reference_feature = reference_feature

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'fit':
        return self

    def transform(self, X: pd.DataFrame) -> 'transform':
        X = X.copy()
        for feature in self.features:
            X[feature] = X[self.reference_feature] - X[feature]
        return X


class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):
    """Replace Rare labels to text:Rare"""

    def __init__(self, tol=0.05, features=None) -> None:
        self.tol = tol
        if not isinstance(features, list):
            self.features = [features]
        else:
            self.features = features
        self.encoder_dict_ = {}

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'fit':
        for feature in self.features:
            tmp = X[feature].value_counts(normalize=True)
            self.encoder_dict_[feature] = list(tmp[tmp > self.tol].index)
        return self

    def transform(self, X: pd.DataFrame) -> 'transform':
        X = X.copy()
        for feature in self.features:
            X[feature] = np.where(
                X[feature].isin(self.encoder_dict_[feature]),
                X[feature],
                'Rare')
        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Replace Categorical Values to Label-Mean ordered Int"""

    def __init__(self, features=None) -> None:
        if not isinstance(features, list):
            self.features = [features]
        else:
            self.features = features
        self.encoder_dict_ = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'fit':
        tmp = pd.concat([X, y], axis=1)
        tmp.columns = list(X.columns) + ['target']

        for feature in self.features:
            t = tmp.groupby([feature])['target'].mean().sort_values().index
            self.encoder_dict_[feature] = {k: i for i, k in enumerate(t, 0)}
            # np.save('./dataset/OrdinalLabels.npy', self.encoder_dict_) # 백업 반영시
        return self

    def transform(self, X: pd.DataFrame) -> 'transform':
        X = X.copy()
        for feature in self.features:
            X[feature] = X[feature].map(self.encoder_dict_[feature])

        return X


class DropUnnecessaryFeatures(BaseEstimator, TransformerMixin):
    """Remove Un-necessary Features"""

    def __init__(self, features=None) -> None:
        if not isinstance(features, list):
            self.features = [features]
        else:
            self.features = features

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'fit':
        return self

    def transform(self, X: pd.DataFrame) -> 'transform':
        X = X.copy()
        X = X.drop(self.features, axis=1)
        return X


class LogTransformer(BaseEstimator, TransformerMixin):
    """Conduct Log transformation"""

    def __init__(self, features=None) -> None:
        if not isinstance(features, list):
            self.features = [features]
        else:
            self.features = features

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'fit':
        return self

    def transform(self, X: pd.DataFrame) -> 'transform':
        X[self.features] = np.log(X[self.features])
        return X


class CustomMinMaxScaler(BaseEstimator, TransformerMixin):
    """Custom MinMaxScaler Basically 0~1"""

    def __init__(self, features=None) -> None:
        if not isinstance(features, list):
            self.features = [features]
        else:
            self.features = features

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'fit':
        return self

    def transform(self, X: pd.DataFrame) -> 'transform':
        X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        return X
