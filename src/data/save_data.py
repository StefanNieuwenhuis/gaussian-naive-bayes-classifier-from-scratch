import pandas as pd

from pathlib import Path

from typing import Self
from numpy.typing import NDArray


class SaveData():
    """
    Save preprocessed data to CSV
    """

    def __init__(self, save_path: str = 'data/preprocessed/' ):
        self.save_path_ = Path(save_path)

    def save(self, X: NDArray, y: NDArray, name: str) -> Self:
        """
        Transform feature matrix by applying fitted IQR bounds

        Parameters
        ----------
        X: nd.array of shape (n_samples, n_features)
            Feature matrix
        y: nd.array of shape (n_samples,)
            Target vector
        name: str
            Identifier string like 'train' or 'test'

        Returns
        -------
        self: class-instance
        """
        X_path = self.save_path_ / f"X_{name}.csv"
        y_path = self.save_path_ / f"y_{name}.csv"

        if X_path.exists():
            raise FileExistsError(f"File {X_path} already exists. Aborting to avoid overwrite.")

        if y_path.exists():
            raise FileExistsError(f"File {y_path} already exists. Aborting to avoid overwrite.")

        # Ensure the parent directories exist
        X_path.parent.mkdir(parents=True, exist_ok=True)
        y_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        pd.DataFrame(X).to_csv(X_path)
        pd.DataFrame(y).to_csv(y_path)

        return self