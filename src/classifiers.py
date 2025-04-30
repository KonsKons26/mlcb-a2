import os

import numpy as np

import pandas as pd

from src.preprocessing import fill_nans_with_median

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

import optuna
from optuna.samplers import TPESampler
from optuna.distributions import (
    IntDistribution, FloatDistribution, CategoricalDistribution
)

from datetime import datetime


def time():
    y = datetime.now().year
    m = datetime.now().month
    d = datetime.now().day
    h = datetime.now().hour
    mi = datetime.now().minute
    s = datetime.now().second
    return f"{y}/{m}/{d}-{h}:{mi}:{s}"


class Classifier:

    # I like having them here, I know it's not needed or anything
    VALID_MODELS = [
        "LR",   # LogisticRegression
        "GNB",  # GaussianNB
        "LDA",  # LinearDiscriminantAnalysis
        "SVM",  # SupportVectorMachine
        "RF",   # RandomForest
        "LGBM"  # LightGBM
    ]

    def __init__(
            self,
            model_type: str,
            dataset: pd.DataFrame,
            target: pd.Series,
            models_dir: str,
            nCV_rounds: int = 10,
            outer_folds: int = 5,
            inner_folds: int = 3,
            fill_nans: bool = True,
            random_state_base: int = 42,
        ):

        # Checks
        if model_type not in self.VALID_MODELS:
            raise ValueError(f"Invalid model type: {model_type}. "
                             f"Valid options are: {self.VALID_MODELS}")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            self.simple_log += f"Directory {models_dir} created."

        self.model_type = model_type
        self.X = dataset.drop(columns=[target])
        self.y = dataset[target]
        self.nCV_rounds = nCV_rounds
        self.outer_folds = outer_folds
        self.inner_folds = inner_folds
        self.models_dir = models_dir
        self.fill_nans = fill_nans
        self.random_state_base = random_state_base
        self.random_states = [
            self.random_state_base + i
            for i in range(self.nCV_rounds)
        ]

        self.simple_log = ""
        self.simple_log += f"\n[{time()}] Classifier class created\n\n"
        self.simple_log += "-"* 50 + "\n"
        self.simple_log += f"| Model type: {self.model_type:>34} |\n"
        self.simple_log += f"| Dataset shape: {str(self.X.shape):>31} |\n"
        self.simple_log += f"| Target shape: {str(self.y.shape):>32} |\n"
        self.simple_log += f"| Number of CV rounds: {self.nCV_rounds:>25} |\n"
        self.simple_log += f"| Outer folds: {self.outer_folds:>33} |\n"
        self.simple_log += f"| Inner folds: {self.inner_folds:>33} |\n"
        self.simple_log += f"| Random state base: {self.random_state_base:>27} |\n"
        self.simple_log += "-"* 50 + "\n\n"

        if fill_nans:
            self.X = fill_nans_with_median(self.X, self.y, class_based=True)
            self.simple_log += f"[{time()}] NaN values filled with median.\n"
        else:
            self.simple_log += f"[{time()}] NaN values not filled.\n"

        self.hyperparameter_space = self._get_hyperparameter_space()
        self.simple_log += f"[{time()}] Hyperparameter space created.\n"

    def _get_hyperparameter_space(self):
        """Returns the hyperparameter space for the model type."""
        if self.model_type == "LR":
            return {
                "penalty": CategoricalDistribution(["elasticnet"]),
                "C": FloatDistribution(1e-5, 1e5, log=True),
                "fit_intercept": CategoricalDistribution([True, False]),
                "solver": CategoricalDistribution(["lbfgs", "liblinear", "sag", "saga"]),
                "random_state": self.random_state,
            }
        elif self.model_type == "GNB":
            return {}
        elif self.model_type == "LDA":
            return {
                "solver": CategoricalDistribution(["svd", "lsqr", "eigen"]),
                "shrinkage": CategoricalDistribution([None, 'auto'])
            }
        elif self.model_type == "SVM":
            return {
                "C": FloatDistribution(1e-5, 1e5, log=True),
                "kernel": CategoricalDistribution(["linear", "poly", "rbf", "sigmoid"]),
                "degree": IntDistribution(2, 7),
                "gamma": CategoricalDistribution(["scale", "auto"]),
                "random_state": self.random_state,
            }
        elif self.model_type == "RF":
            return {
                "n_estimators": IntDistribution(10, 1000),
                "criterion": CategoricalDistribution(["gini", "entropy", "log_loss"]),
                "max_depth": IntDistribution(10, 50),
                "min_samples_split": IntDistribution(2, 20),
                "min_samples_leaf": IntDistribution(1, 20),
                "max_features": CategoricalDistribution(["sqrt", "log2"]),
                "random_state": self.random_state
            }
        elif self.model_type == "LGBM":
            return {
                "boosting_type": CategoricalDistribution(["gbdt", "dart"]),
                "num_leaves": IntDistribution(20, 150),
                "learning_rate": FloatDistribution(1e-3, 0.5),
                "n_estimators": IntDistribution(50, 500)
            }
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")


if __name__ == "__main__":
    test = Classifier(
        model_type="LR",
        dataset=pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [5, 4, 3, 2, 1],
            "target": [0, 1, 0, 1, 0]
        }),
        target="target",
        models_dir="./models"
    )
    print(test.simple_log)