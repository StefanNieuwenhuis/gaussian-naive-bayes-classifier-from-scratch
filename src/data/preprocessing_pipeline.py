from typing import Self

from exceptiongroup import catch
from numpy.typing import NDArray

from data.base import Transformer


class PreProcessingPipeline(Transformer):
    """
    Preprocessing pipeline to clean and pre-process raw data
    """

    def __init__(self, steps: list[Transformer]):
        super().__init__()
        self.steps = steps or []

        self.logger.info(f"Initialized PreprocessingPipeline with {len(self.steps)} steps")

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
        self.logger.info(f"Starting pipeline fit")

        for idx, step in enumerate(self.steps):
            try:
                self.logger.info(f"Fitting step {idx + 1}/{len(self.steps)}: {step.__class__.__name__}")
                step.fit(X, y)
                self.logger.info(f"Step {step.__class__.__name__} fit complete")
            except Exception as e:
                self.logger.error(f"Error fitting {step.__class__.__name__}: {e}")
                raise

        self.logger.info("Pipeline fit complete")
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
        self.logger.info("Starting pipeline transform")

        for i, step in enumerate(self.steps):
            try:
                self.logger.info(f"Transforming step {i + 1}/{len(self.steps)}: {step.__class__.__name__}")
                if y is not None:
                    X, y = step.transform(X, y)
                else:
                    X = step.transform(X)
                self.logger.info(f"Step {step.__class__.__name__} transform complete")
            except Exception as e:
                self.logger.error(f"Error transforming {step.__class__.__name__}: {e}")
                raise

        self.logger.info("Pipeline transform complete")

        return (X, y) if y is not None else (X, None)

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

        self.logger.info("Starting pipeline fit_transform")
        return self.fit(X,y).transform(X, y)