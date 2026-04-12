from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureBuilderTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn wrapper around existing FeatureBuilder.

    FeatureBuilder has:
      - fit(df, y=None) -> self
      - transform(df) -> (pd.DataFrame)
    """
    def __init__(self, feature_builder: Any, return_numpy: bool = True):
        self.feature_builder = feature_builder
        self.return_numpy = return_numpy
        self.feature_names = None

    def fit(self, X: pd.DataFrame):
        # Fit only on training data
        self.feature_builder.fit(X)
        return self

    def transform(self, X: pd.DataFrame):
        df = self.feature_builder.transform(X)
        # Return DataFrame or numpy
        if isinstance(df, pd.DataFrame):
            self.feature_names = list(df.columns)
        if self.return_numpy:
            return df.to_numpy(dtype=float)
        return df

    # helps with debugging and pipelines
    def get_feature_names_out(self, input_features=None):
        if hasattr(self, "feature_names"):
            return np.array(self.feature_names, dtype=object)
        return np.array([], dtype=object)
