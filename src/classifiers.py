import os

import numpy as np

import pandas as pd

from src.preprocessing import fill_nans_with_median

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from lightgbm import LGBMClassifier

from mrmr import mrmr_classif

import optuna
from optuna.samplers import TPESampler
from optuna.distributions import IntDistribution, FloatDistribution, CategoricalDistribution

import joblib

from datetime import datetime

VALID_MODELS = {
    "LogisticRegression": LogisticRegression(),
    "GaussianNB": GaussianNB(),
    "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis(),
    "SVC": SVC(),
    "RandomForestClassifier": RandomForestClassifier(),
    "LGBMClassifier": LGBMClassifier()
}


def time():
    """Returns the current time in a formatted string."""
    now = datetime.now()
    return f"{now:%Y/%m/%d-%H:%M:%S}"


class NestedCrossValidation:
    def __init__(self, classifier, X, y,
                 n_rounds=10, n_outer_folds=5, n_inner_folds=3,
                 n_features_to_select=5, n_optuna_trials=50,
                 metric=matthews_corrcoef, random_state_base=42):
        self.classifier = VALID_MODELS[classifier]
        self.X = X
        self.y = y
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
        """Define hyperparameter spaces for each classifier."""
        return {
            "LogisticRegression": lambda trial: {
                "solver": "saga",
                "penalty": "elasticnet",
                "C": trial.suggest_float("C", 1e-5, 1e5, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
                "max_iter": trial.suggest_int("max_iter", 100, 10_000),
                "class_weight": trial.suggest_categorical("class_weight", ["balanced", None])
            },
            "GaussianNB": lambda trial: {
                "var_smoothing": trial.suggest_float("var_smoothing", 1e-10, 1e-5, log=True)
            },
            "LinearDiscriminantAnalysis": lambda trial: {
                "solver": trial.suggest_categorical("solver", ["svd", "lsqr", "eigen"]),
                "shrinkage": trial.suggest_categorical("shrinkage", [None, "auto"]),
                "priors": trial.suggest_categorical("priors", [None, [0.5, 0.5]])
            },
            "SVC": lambda trial: {
                "C": trial.suggest_float("C", 1e-5, 1e5, log=True),
                "gamma": trial.suggest_float("gamma", 1e-5, 1e5, log=True),
                "kernel": trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"]),
                "degree": trial.suggest_int("degree", 1, 5) if trial.params.get("kernel") == "poly" else 3,
                "class_weight": trial.suggest_categorical("class_weight", ["balanced", None])
            },
            "RandomForestClassifier": lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
                "max_depth": trial.suggest_int("max_depth", 1, 100),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None])
            },
            "LGBMClassifier": lambda trial: {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
                "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1.0, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 10, 100),
                "max_depth": trial.suggest_int("max_depth", 1, 100),
                "min_child_samples": trial.suggest_int("min_child_samples", 1, 100),
                "subsample": trial.suggest_float("subsample", 0.1, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
                "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
                "verbose": -1
            }
        }

    def _objective(self, trial, X_train, y_train, X_val, y_val):
        """Objective function for Optuna hyperparameter tuning."""
        # Get hyperparameters for the current trial
        hyperparams = self.hyperparam_spaces[self.classifier](trial)

        # Create the model with the selected hyperparameters
        model = self.classifier.set_params(**hyperparams)
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)

        # Calculate the metric
        return self.metric(y_val, y_val_pred)


    def run(self):
        self.results = []

        for round_ in range(self.n_rounds):
            round_seed = self.random_state_base + round_

            outer_cv = StratifiedKFold(n_splits=self.n_outer_folds, shuffle=True, random_state=round_seed)

            # Enter outer loop - Cross Validation
            for outer_train_idx, outer_test_idx in outer_cv.split(self.X, self.y):
                X_outer_train, X_outer_test = self.X.iloc[outer_train_idx], self.X.iloc[outer_test_idx]
                y_outer_train, y_outer_test = self.y.iloc[outer_train_idx], self.y.iloc[outer_test_idx]

                # Enter inner loop - Hyperparameter tuning
                inner_cv = StratifiedKFold(n_splits=self.n_inner_folds, shuffle=True, random_state=round_seed)
                inner_best_score = -np.inf
                for inner_train_idx, inner_val_idx in inner_cv.split(X_outer_train, y_outer_train):
                    X_inner_train, X_inner_val = X_outer_train.iloc[inner_train_idx], X_outer_train.iloc[inner_val_idx]
                    y_inner_train, y_inner_val = y_outer_train.iloc[inner_train_idx], y_outer_train.iloc[inner_val_idx]

                    # Normalize the data
                    scaler = StandardScaler()
                    scaler.fit(X_inner_train)
                    X_inner_train = pd.DataFrame(scaler.transform(X_inner_train), columns=X_inner_train.columns)
                    X_inner_val = pd.DataFrame(scaler.transform(X_inner_val), columns=X_inner_val.columns)

                    # Feature selection using mRMR
                    selected_features = mrmr_classif(X_inner_train, y_inner_train, n=self.n_features_to_select)
                    X_inner_train = X_inner_train[selected_features]
                    X_inner_val = X_inner_val[selected_features]

                    # Optuna study for hyperparameter tuning
                    study = optuna.create_study(sampler=TPESampler(seed=round_seed), direction="maximize")
                    study.optimize(lambda trial: self._objective(trial, X_inner_train, y_inner_train, X_inner_val, y_inner_val), n_trials=self.n_optuna_trials)

                    trial_score = study.best_value
                    if trial_score > inner_best_score:
                        inner_best_score = trial_score
                        best_trial = study.best_trial
                        best_hyperparams = best_trial.params

                # Train the model with the best hyperparameters on the outer training set
                model = self.classifier.set_params(**best_hyperparams)
                model.fit(X_outer_train, y_outer_train)
                y_outer_pred = model.predict(X_outer_test)

                # Calculate metrics
                accuracy = accuracy_score(y_outer_test, y_outer_pred)
                precision = precision_score(y_outer_test, y_outer_pred)
                recall = recall_score(y_outer_test, y_outer_pred)
                f1 = f1_score(y_outer_test, y_outer_pred)
                mcc = matthews_corrcoef(y_outer_test, y_outer_pred)
                self.results.append({
                    "round": round_,
                    "outer_fold": outer_train_idx,
                    "inner_best_score": inner_best_score,
                    "best_hyperparams": best_hyperparams,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "mcc": mcc
                })
        self.summary = pd.DataFrame(self.results)
        self.summary["mean_accuracy"] = self.summary.groupby("round")["accuracy"].transform("mean")
        self.summary["std_accuracy"] = self.summary.groupby("round")["accuracy"].transform("std")
        self.summary["mean_precision"] = self.summary.groupby("round")["precision"].transform("mean")
        self.summary["std_precision"] = self.summary.groupby("round")["precision"].transform("std")
        self.summary["mean_recall"] = self.summary.groupby("round")["recall"].transform("mean")
        self.summary["std_recall"] = self.summary.groupby("round")["recall"].transform("std")
        self.summary["mean_f1"] = self.summary.groupby("round")["f1"].transform("mean")
        self.summary["std_f1"] = self.summary.groupby("round")["f1"].transform("std")
        self.summary["mean_mcc"] = self.summary.groupby("round")["mcc"].transform("mean")
        self.summary["std_mcc"] = self.summary.groupby("round")["mcc"].transform("std")