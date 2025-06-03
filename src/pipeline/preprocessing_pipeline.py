from typing import Self
from numpy.typing import NDArray

from data.base import Transformer


class PreProcessingPipeline(Transformer):
    """
    Preprocessing pipeline to clean and pre-process raw data
    """

    def __init__(self, steps: list[Transformer]):
        self.steps = steps

    def fit(self, X: NDArray, y: NDArray) -> Self:
        """
        Fit all transformers

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

        for step in self.steps:
            step.fit(X, y)
        return self

    def transform(self, X: NDArray, y: NDArray) -> NDArray:
        """
        Transform all transformers

        Parameters
        ----------
        X: nd.array of shape (n_samples, n_features)
            Feature matrix
        y: nd.array of shape (n_samples,)
            Target vector

        Returns
        -------
        C: tuple of nd.array
            Test and training feature matrices, and
            test and training target vectors
        """
        for step in self.steps:
            X, y = step.transform(X, y)

        return X, y

    def fit_transform(self, X: NDArray, y: NDArray) -> NDArray:
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