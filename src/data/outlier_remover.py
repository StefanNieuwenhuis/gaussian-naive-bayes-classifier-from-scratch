from typing import Self

import numpy as np
from numpy.typing import NDArray

from data.base import Transformer


class OutlierRemover(Transformer):
    """
    Remove Outliers from a given Feature Matrix over unique class labels
    """

    def __init__(self, threshold = 1.5):
        super().__init__()
        self.threshold_ = threshold
        self.classes_ = None
        self.q1_ = None
        self.q3_ = None

    def fit(self, X: NDArray, y: NDArray=None) -> Self:
        """
        Learn the IQR bounds for the given Feature matrix

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
        if len(X) == 0:
            self.logger.error("Feature matrix X is None.")
            raise ValueError("OutlierRemover requires a Feature matrix to learn IQR bounds")

        if len(y) == 0:
            self.logger.error("Target vector y is None.")
            raise ValueError("OutlierRemover requires a target column (e.g., 'species') for group-wise IQR.")

        self.logger.info("Fitting OutlierRemover with threshold=%.2f", self.threshold_)

        self.classes_, class_indices = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.q1_ = np.zeros((n_classes, n_features))
        self.q3_ = np.zeros((n_classes, n_features))

        for class_id in range(n_classes):
            subset = X[class_indices == class_id]

            self.q1_[class_id] = np.percentile(subset, 25, axis=0)
            self.q3_[class_id] = np.percentile(subset, 75, axis=0)

        self.logger.debug("Q1 shape: %s, Q3 shape: %s", self.q1_.shape, self.q3_.shape)
        return self

    def transform(self, X: NDArray, y: NDArray) -> NDArray:
        """
        Transform feature matrix by applying fitted IQR bounds

        Parameters
        ----------
        X: nd.array of shape (n_samples, n_features)
            Feature matrix
        y: nd.array of shape (n_samples,)
            Target vector

        Returns
        -------
        C: nd.array of shape (n_samples, n_features)
            Transformed feature matrix
        """
        if self.q1_ is None or self.q3_ is None:
            self.logger.error("OutlierRemover instance is not fitted")
            raise ValueError("OutlierRemover instance is not fitted yet. Call 'fit' first.")

        self.logger.info("Transforming with fitted IQR bounds")

        # Map y labels to class indices (0...n_classes-1)
        class_indices = np.searchsorted(self.classes_, y)

        # Compute Inter-Quartile Range
        Q1 = self.q1_[class_indices]
        Q3 = self.q3_[class_indices]
        IQR = Q3 - Q1

        # Compute upper & lower bounds
        lower_bound = Q1 - self.threshold_ * IQR
        upper_bound = Q3 + self.threshold_ * IQR

        # Identify outliers: True where outlier in any feature
        is_outlier = (X < lower_bound) | (X > upper_bound)
        mask = ~np.any(is_outlier, axis=1)

        self.logger.debug("Returning %d samples after outlier removal", len(X[mask]))

        return X[mask], y[mask]

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
            Transformed feature matrix
        """
        return self.fit(X, y).transform(X, y)
