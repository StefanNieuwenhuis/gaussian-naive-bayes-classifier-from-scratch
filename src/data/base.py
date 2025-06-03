from abc import abstractmethod
from typing import Self
from numpy.typing import NDArray


class Transformer:
    @abstractmethod
    def fit(self, X: NDArray, y: NDArray=None) -> Self:
        """
        Abstract method that learns and stores any statistics or parameters needed for transformation.

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

    @abstractmethod
    def transform(self, X: NDArray, y: NDArray) -> NDArray:
        """
        Abstract method to applies fitted transformation to the dataset provided.

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

    def fit_transform(self, X: NDArray, y=None)-> NDArray:
        """
        Convenience method that applies fit and trasnform in a single call.

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

        self.fit(X, y)

        return self.transform(X)