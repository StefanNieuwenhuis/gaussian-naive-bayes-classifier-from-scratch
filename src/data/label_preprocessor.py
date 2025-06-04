from typing import Self

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import LabelEncoder

from data.base import Transformer


class LabelPreProcessor(Transformer):
    """
    Encodes target vector labels
    """

    def __init__(self):
        super().__init__()
        self.encoder_ = LabelEncoder()

    def fit(self, X: NDArray, y: NDArray = None) -> Self:
        """
        Learn the unique classes in the dataset provided

        Parameters
        ----------
        X: nd.array of shape (n_samples, n_features)
            Feature matrix
        y: nd.array of shape (n_samples,)
            Target vector

        Returns
        -------
        self: class-instance
        """

        if y is None:
            error_message = "LabelPreprocessor requires `y` to be non-null."
            self.logger.error(error_message)
            raise ValueError(error_message)

        self.logger.info("Fitting LabelPreprocessor on %d unique classes", len(np.unique(y)))
        self.encoder_.fit(y)

        return self

    def transform(self, X: NDArray, y: NDArray) -> NDArray:
        """
        Transform target vector by applying fitted class labels

        Parameters
        ----------
        X: nd.array of shape (n_samples, n_features)
            Feature matrix

        Returns
        -------
        C: nd.array of shape (n_samples,)
            Transformed target vector
        """
        self.logger.info("Transforming label classes on %d unique classes", len(np.unique(y)))
        return X, self.encoder_.transform(y)

    def fit_transform(self, X: NDArray, y=None) -> NDArray:
        """
        Apply fit and transform in a single call

        Parameters
        ----------
        X: nd.array of shape (n_samples, n_features)
            Feature matrix
        y: nd.array of shape (n_samples,)
            Target vector

        Returns
        -------
        C: nd.array of shape (n_samples, n_features)
            Transformed target vector
        """

        return self.fit(X, y).transform(X, y)

    def inverse_transform(self, y_encoded: NDArray) -> NDArray:
        """
        Inverse-Transform target vector to original class labels

        Parameters
        ----------
        y: nd.array of shape (n_samples,)
            Encoded Target vector

        Returns
        -------
        C: nd.array of shape (n_samples, n_features)
            Inverse-Transformed target vector
        """
        self.logger.info("Inverse-transforming encoded class labels on %d unique classes", len(np.unique(y_encoded)))
        return self.encoder_.inverse_transform(y_encoded)