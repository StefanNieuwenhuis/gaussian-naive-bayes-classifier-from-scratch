from abc import abstractmethod, ABC
from typing import Self
from numpy.typing import NDArray

from utils.logging_factory import LoggerFactory



class BaseTransformer(ABC):
    """
    Base transformer class
    """
    def __init__(self):
        self.logger = LoggerFactory.get_logger(self.__class__.__name__)

class StatelessTransformer(BaseTransformer):
    """
    Stateless Transformer base class
    """

    def __init__(self):
        super().__init__()

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

class Transformer(BaseTransformer):
    """
    Transformer base class
    """

    def __init__(self):
        super().__init__()

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
        Convenience method that applies fit and transform in a single call.

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

        return self.fit(X,y).transform(X, y)
