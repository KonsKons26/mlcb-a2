import os

import numpy as np

import pandas as pd

from src.preprocessing import fill_nans_with_median

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from lightgbm import LGBMClassifier

from mrmr import mrmr_classif

import optuna
from optuna.samplers import TPESampler
from optuna.distributions import (
    IntDistribution, FloatDistribution, CategoricalDistribution
)

import joblib

from datetime import datetime

VALID_MODELS = {
    "LogisticRegression":   LogisticRegression(),
    "GaussianNB":  GaussianNB(),
    "LinearDiscriminantAnalysis":  LinearDiscriminantAnalysis(),
    "SVC":  SVC(),
    "RandomForestClassifier":   RandomForestClassifier(),
    "LGBMClassifier": LGBMClassifier()
}


def time():
    """Returns the current time in a formatted string."""
    now = datetime.now()
    return f"{now:%Y/%m/%d-%H:%M:%S}"


class NestedCrossValidation:
    def __init__(self, X, y, classifier,
                 n_rounds=10, n_outer_folds=5, n_inner_folds=3,
                 n_features_to_select=5, n_optuna_trials=50,
                 metric=matthews_corrcoef, random_state_base=42):
        self.X = X
        self.y = y
        self.classifier = classifier
        self.n_rounds = n_rounds
        self.n_outer_folds = n_outer_folds
        self.n_inner_folds = n_inner_folds
        self.n_features_to_select = n_features_to_select
        self.n_optuna_trials = n_optuna_trials
        self.metric = metric
        self.random_state_base = random_state_base
        self.results = []
        self.summary = None

        self.hyperparam_spaces = self._define_hyperparameter_spaces()

    def _define_hyperparameter_spaces(self):
        return {
            "LogisticRegression": lambda trial: {
                "solver": "saga",
                "penalty": "elasticnet",
                "C": trial.suggest_float(
                    "C", 1e-5, 1e5, log=True
                ),
                "l1_ratio": trial.suggest_float(
                    "l1_ratio", 0.0, 1.0
                ),
                "max_iter": trial.suggest_int(
                    "max_iter", 100, 10_000
                ),
                "class_weight": trial.suggest_categorical(
                    "class_weight", ["balanced", None]
                )
            },
            "GaussianNB": lambda trial: {
                "var_smoothing": trial.suggest_float(
                    "var_smoothing", 1e-10, 1e-5, log=True
                )
            },
            "LinearDiscriminantAnalysis": lambda trial: {
                "solver": trial.suggest_categorical(
                    "solver", ["svd", "lsqr", "eigen"]
                ),
                "shrinkage": trial.suggest_categorical(
                    "shrinkage", [None, "auto"]
                ),
                "priors": trial.suggest_categorical(
                    "priors", [None, [0.5, 0.5]]
                )
            },
            "SVC": lambda trial: {
                "C": trial.suggest_float(
                    "C", 1e-5, 1e5, log=True
                ),
                "gamma": trial.suggest_float(
                    "gamma", 1e-5, 1e5, log=True
                ),
                "kernel": trial.suggest_categorical(
                    "kernel", ["linear", "poly", "rbf", "sigmoid"]
                ),
                "degree": trial.suggest_int(
                    "degree", 1, 5
                ) if trial.params.get("kernel") == "poly" else 3,
                "class_weight": trial.suggest_categorical(
                    "class_weight", ["balanced", None]
                )
            },
            "RandomForestClassifier": lambda trial: {
                "n_estimators": trial.suggest_int(
                    "n_estimators", 10, 1000
                ),
                "max_depth": trial.suggest_int(
                    "max_depth", 1, 100
                ),
                "min_samples_split": trial.suggest_int(
                    "min_samples_split", 2, 20
                ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf", 1, 20
                ),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", None]
                ),
                "class_weight": trial.suggest_categorical(
                    "class_weight", ["balanced", "balanced_subsample", None]
                )
            },
            "LGBMClassifier": lambda trial: {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": trial.suggest_categorical(
                    "boosting_type", ["gbdt", "dart"]
                ),
                "n_estimators": trial.suggest_int(
                    "n_estimators", 10, 1000
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-5, 1.0, log=True
                ),
                "num_leaves": trial.suggest_int(
                    "num_leaves", 10, 100
                ),
                "max_depth": trial.suggest_int(
                    "max_depth", 1, 100
                ),
                "min_child_samples": trial.suggest_int(
                    "min_child_samples", 1, 100
                ),
                "subsample": trial.suggest_float(
                    "subsample", 0.1, 1.0
                ),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.1, 1.0
                ),
                "reg_alpha": trial.suggest_float(
                    "reg_alpha", 0.0, 1.0
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", 0.0, 1.0
                ),
                "class_weight": trial.suggest_categorical(
                    "class_weight", ["balanced", None]
                ),
                "verbose": -1
            }
        }