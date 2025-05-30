# Libraries
import numpy as np

# Typings
from typing import Self, Any
from numpy.typing import NDArray

class GaussianNaiveBayes:
    """
    A Gaussian Naive Bayes Classifier

    Implements Bayes' Theorem assuming feature
    independence and Gaussian likelihoods.
    """

    def __init__(self):
        self.classes_: NDArray[Any] = None
        self.class_count_: NDArray[np.int64] = None
        self.class_prior_: NDArray[np.float64] = None
        self.class_curvature_: NDArray[np.float64] = None
        self.class_mean_pull_: NDArray[np.float64] = None
        self.class_log_likelihood_consts_: NDArray[np.float64] = None

    def __compute_log_prior_probabilities(self, y: NDArray) -> Self:
        """
        Computes log prior probabilities for each class

        Parameters
        ----------
        y: nd.array of shape (n_samples,)
            Target vector

        Returns
        -------
        self: class-instance
        """

        # Store unique classes and n_samples_per_class in class attributes
        self.classes_, self.class_count_ = np.unique(y, return_counts=True)

        # Compute log prior probabilities
        self.class_prior_ = np.log(self.class_count_ / self.class_count_.sum())

        return self

    def __compute_log_likelihood_constants(self, X: NDArray, y: NDArray, epsilon: np.int64) -> Self:
        """
        Compute log-likelihood constants for feature matrix X

        Parameters
        ----------
        X: nd.array of shape (n_sample, n_features)
            Feature matrix
        y: nd.array of shape (n_samples,)
            Target vector
        epsilon: np.int64
            variance smoothing value to prevent division by zero, and provide numerical stability

        Returns
        -------
        self: class-instance
        """

        # First, we have to group the feature matrix by class
        # We use a boolean mask, a matrix (shape: n_classes x n_features)
        # mask[i,j] = True if sample j belongs to class i
        mask = (y == self.classes_[:, None])

        # Reshape class_count_ class attribute from shape: (n_classes, ) to shape: (n_classes, 1)
        # Required for matrix multiplication below
        counts = self.class_count_[:, None]

        # Compute per-class sums of features (shape: n_classes x n_features)
        sums = mask @ X

        # Compute per-class means
        thetas = sums / counts

        # Compute sums of squares for variance computation
        X_squared = X ** 2

        # Compute per-class sums of squares
        sums_of_squares = mask @ X_squared

        # Compute variances = E[x^2] - (E[x])^2 + ϵ
        variances = (sums_of_squares / counts) - (thetas ** 2) + epsilon

        # With thetas and variances computed, we can pre-compute the log-likelihood constants
        self.class_curvature_ = -0.5 / variances
        self.class_mean_pull_ = thetas / variances
        self.class_log_likelihood_consts_ = -0.5 * np.log(2 * np.pi * variances) - (thetas ** 2) / (2 * variances)

        return self

    def __compute_joint_log_likelihood(self, X: NDArray) -> NDArray:
        """
        Compute joint log likelihood

        Parameters
        ----------
        X: nd.array of shape (n_sample, n_features)
            Feature matrix

        Returns
        -------
        C: nd.array of shape (n_classes, )
            Classification results vector
        """

        # Compute log-likelihood terms
        log_likelihood = (
                (X ** 2) @ self.class_curvature_.T +
                X @ self.class_mean_pull_.T +
                np.sum(self.class_log_likelihood_consts_, axis=1)
        )

        # Add prior (log) probabilities to each row
        joint_log_likelihood = log_likelihood + self.class_prior_

        return joint_log_likelihood

    def fit(self, X: NDArray, y: NDArray, epsilon=1e-9) -> Self:
        """
        Fits the Gaussian Naive Bayes model
        according to the given training data

        Parameters
        ----------
        X: nd.array of shape (n_sample, n_features)
            Feature matrix
        y : nd.array of shape (n_samples,)
            Target vector
        epsilon: np.int64
            variance smoothing value to prevent division by zero, and provide numerical stability

        Returns
        -------
        self: class-instance
        """

        # Always check if the provided data meets the most basic needs.
        # In our case: Both X and y cannot be empty
        if len(X) == 0:
            raise ValueError("Feature Matrix cannot be empty")

        if len(y) == 0:
            raise ValueError("Target Vector cannot be empty")

        # Compute prior probabilities
        self.__compute_log_prior_probabilities(y)

        # Compute log-likelihood constants for feature matrix X, and target vector y
        self.__compute_log_likelihood_constants(X, y, epsilon)

        return self

    def predict(self, X: NDArray) -> NDArray:
        """
        Perform classification on an array of test vectors X

        Parameters
        ----------
        X: nd.array of shape (n_sample, n_features)
            Feature matrix

        Returns
        -------
        C: nd.array of shape (n_classes, )
            Classification results vector
        """
        joint_log_likelihood = self.__compute_joint_log_likelihood(X)

        return np.argmax(joint_log_likelihood, axis=1)

    def predict_log_proba(self, X: NDArray) -> NDArray:
        """
        Estimate log probability outputs on an array of test vectors

        Parameters
        ----------
        X: nd.array of shape (n_sample, n_features)
            Feature matrix

        Returns
        -------
        C: nd.array of shape (n_samples, n_classes)
            Returns the log probability of the samples for each class in
            the model
        """

        # Compute joint log likelihood
        joint_log_likelihood = self.__compute_joint_log_likelihood(X)

        # Apply log-sum-exp transform
        max_joint_log_likelihood = np.max(joint_log_likelihood, axis=1, keepdims=True)
        logsumexp = max_joint_log_likelihood + np.log(
            np.sum(np.exp(joint_log_likelihood - max_joint_log_likelihood), axis=1, keepdims=True))

        return joint_log_likelihood - logsumexp

    def predict_proba(self, X: NDArray) -> NDArray:
        """
        Return probability estimates for the test vector X.

        Parameters
        ----------
        X: nd.array of shape (n_sample, n_features)
            Feature matrix

        Returns
        -------
        C : nd.array of shape (n_samples, n_classes)
            Returns the probability of the samples for each class in
            the model
        """

        return np.exp(self.predict_log_proba(X))