from typing import Tuple

from numpy.typing import NDArray
from sklearn.model_selection import train_test_split



class DataSplitter:
    """
    Split feature matrix and target vector into training and testing datasets
    """

    def __init__(self, test_size = 0.2, stratify = None, random_state = 42):
        super().__init__()
        self.test_size = test_size
        self.stratify = stratify
        self.random_state = random_state

    def split(self, X: NDArray, y: NDArray) -> Tuple[NDArray, ...]:
        """
        Split feature matrix and target vector into training and test datasets

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
            Order: X_train, X_test, y_train, y_test
        """
        return train_test_split(
            X,
            y,
            test_size=self.test_size,
            stratify=self.stratify,
            random_state=self.random_state
        )
