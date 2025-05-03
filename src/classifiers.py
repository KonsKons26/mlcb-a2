import os

import numpy as np

import pandas as pd

from src.preprocessing import fill_nans_with_median

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    matthews_corrcoef, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
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

from joblib import dump, load

from datetime import datetime

optuna.logging.set_verbosity(optuna.logging.WARNING)


VALID_MODELS = {
    "LogisticRegression": LogisticRegression(),
    "GaussianNB": GaussianNB(),
    "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis(),
    "SVC": SVC(),
    "RandomForestClassifier": RandomForestClassifier(),
    "LGBMClassifier": LGBMClassifier(verbosity=-1)
}


def time():
    """
    Returns the current time in a formatted string.
    """
    now = datetime.now()
    return f"{now:%Y/%m/%d-%H:%M:%S}"


def specificity_score(y_true, y_pred):
    """Calculate specificity score."""
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


class NestedCrossValidation:
    def __init__(self, classifier, X, y, models_dir, results_dir,
                 n_rounds=10, n_outer_folds=5, n_inner_folds=3,
                 n_features_to_select=5, n_optuna_trials=50,
                 metric=matthews_corrcoef, random_state_base=42):
        """
        Initialize the NestedCrossValidation class.

        Parameters
        ----------
        classifier : str
            The name of the classifier to use. Must be one of the keys in
            VALID_MODELS.
        X : pd.DataFrame
            The feature matrix.
        y : pd.Series
            The target variable.
        models_dir : str
            The directory where the models will be saved.
        results_dir : str
            The directory where the results will be saved.
        n_rounds : int, default=10
            The number of rounds for cross-validation.
        n_outer_folds : int, default=5
            The number of outer folds for cross-validation.
        n_inner_folds : int, default=3
            The number of inner folds for cross-validation.
        n_features_to_select : int, default=5
            The number of features to select using mRMR.
        n_optuna_trials : int, default=50
            The number of trials for Optuna hyperparameter tuning.
        metric : callable, default=matthews_corrcoef
            The metric to use for evaluating the model.
        random_state_base : int, default=42
            The base random state for reproducibility.
        """
        # User-defined attributes
        self.classifier_type = classifier
        self.classifier = VALID_MODELS[classifier]
        self.X = X
        self.y = y
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.n_rounds = n_rounds
        self.n_outer_folds = n_outer_folds
        self.n_inner_folds = n_inner_folds
        self.n_features_to_select = n_features_to_select
        self.n_optuna_trials = n_optuna_trials
        self.metric = metric
        self.random_state_base = random_state_base

        # Initialize auto-generated attributes
        self.results = []
        self.summary = None
        self.hyperparam_spaces = self._define_hyperparameter_spaces()

    def _define_hyperparameter_spaces(self):
        """
        Define the hyperparameter spaces for each classifier.

        This method returns a dictionary where the keys are classifier
        names and the values are functions that take an Optuna trial
        object and return a dictionary of hyperparameters.
        The hyperparameters are defined using Optuna's suggest methods,
        which allow for flexible and efficient hyperparameter tuning.

        Returns
        -------
        dict
            A dictionary containing the hyperparameter spaces for each
            classifier.
        """
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
                "C": trial.suggest_float("C", 0.1, 1, log=True),
                "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
                "coef0": trial.suggest_float("coef0", 0.0, 1.0),
                "kernel": trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"]),
                "degree": trial.suggest_int(
                    "degree", 1, 3
                ) if trial.params.get("kernel") == "poly" else 3,
                "class_weight": trial.suggest_categorical("class_weight", ["balanced", None])
            },
            "RandomForestClassifier": lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 10, 100),
                "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
                "max_depth": None,
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20)
            },
            "LGBMClassifier": lambda trial: {
                "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart", "rf"]),
                "num_leaves": trial.suggest_int("num_leaves", 10, 100),
                "max_depth": trial.suggest_int("max_depth", 1, 100),
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1.0, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 10, 1_000),
                "objective": "binary",
                "min_child_samples": trial.suggest_int("min_child_samples", 1, 100),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
                "verbose": -1,
                # For rf 
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.1, 0.99),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 0.99)
            }
        }

    def _objective(self, trial, X_train, y_train, X_val, y_val):
        """
        Objective function for Optuna hyperparameter tuning.

        This method defines the objective function that Optuna will
        optimize. It takes an Optuna trial object and the training and
        validation data as input. The function returns the metric score
        for the model trained with the hyperparameters suggested by
        Optuna.

        Parameters
        ----------
        trial : optuna.Trial
            The Optuna trial object.
        X_train : pd.DataFrame
            The training feature matrix.
        y_train : pd.Series
            The training target variable.
        X_val : pd.DataFrame
            The validation feature matrix.
        y_val : pd.Series
            The validation target variable.

        Returns
        -------
        float
            The metric score for the model trained with the suggested
            hyperparameters.
        """
        # Get hyperparameters for the current trial
        hyperparams = self.hyperparam_spaces[self.classifier_type](trial)

        # Fix issues with the hyperparameters
        if self.classifier_type == "LinearDiscriminantAnalysis":
            if hyperparams["solver"] == "svd":
                hyperparams["shrinkage"] = None

        model_class = VALID_MODELS[self.classifier_type].__class__
        model = model_class(**hyperparams)

        # Train the model with the selected hyperparameters
        model.fit(X_train, y_train)

        # Predict on the validation set
        y_val_pred = model.predict(X_val)

        # Calculate the metric
        return self.metric(y_val, y_val_pred)

    def _save_model(self, model, model_name):
        """Saves the model to the models directory."""
        model_name = model_name + ".joblib"
        dump(model, os.path.join(self.models_dir, model_name))

    def _save_scaler(self, scaler, scaler_name):
        """Saves the scaler to the models directory."""
        scaler_name = scaler_name + "_scaler.joblib"
        dump(scaler, os.path.join(self.models_dir, scaler_name))

    def _save_features(self, features, model_name):
        """Saves the features to the models directory."""
        features_name = model_name + "_features.txt"
        with open(os.path.join(self.models_dir, features_name), "w") as f:
            for feature in features:
                f.write(f"{feature}\n")

    def run(self):
        """
        Run the nested cross-validation.

        This method performs the nested cross-validation process, including
        hyperparameter tuning and feature selection. It stores the results
        in the `self.results` attribute and summarizes the results in
        `self.summary`.
        """
        print("="*50)
        print(f"Running Nested Cross Validation for {self.classifier_type}...")
        print(f"Random state base: {self.random_state_base}")
        print(f"Number of rounds: {self.n_rounds}")
        print(f"Number of outer folds: {self.n_outer_folds}")
        print(f"Number of inner folds: {self.n_inner_folds}")
        print(f"Number of features to select: {self.n_features_to_select}")
        print(f"Number of Optuna trials: {self.n_optuna_trials}")
        print("="*50)

        # FS ---------------------------------------------------------------
        selected_features = mrmr_classif(
            X=self.X,
            y=self.y,
            K=self.n_features_to_select,
            show_progress=False
        )

        # --- nCV Rounds -------------------------------------------------------
        for round_ in range(self.n_rounds):
            print()
            print(f"Round {round_ + 1}/{self.n_rounds}...")


            # Set the random seed for this round
            round_seed = self.random_state_base + round_

            # --- Outer loops --------------------------------------------------
            # Cross Validation
            outer_cv = StratifiedKFold(
                n_splits=self.n_outer_folds,
                shuffle=True, random_state=round_seed
            )
            for outer_fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(self.X, self.y)):
                print()
                print(f"Outer fold {outer_fold_idx + 1}/{self.n_outer_folds}...")

                # Set up the data
                X_outer_train, X_outer_test = self.X.iloc[outer_train_idx], self.X.iloc[outer_test_idx]
                y_outer_train, y_outer_test = self.y.iloc[outer_train_idx], self.y.iloc[outer_test_idx]

                # --- Inner loop ----------------------------------------------- 
                # Hyperparameter Tuning
                inner_cv = StratifiedKFold(
                    n_splits=self.n_inner_folds,
                    shuffle=True, random_state=round_seed
                )
                inner_best_score = -np.inf
                best_trial = None
                for inner_fold_idx, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(X_outer_train, y_outer_train)):
                    print(f"\rInner fold {inner_fold_idx + 1}/{self.n_inner_folds}...", end="")
                    X_inner_train, X_inner_val = X_outer_train.iloc[inner_train_idx], X_outer_train.iloc[inner_val_idx]
                    y_inner_train, y_inner_val = y_outer_train.iloc[inner_train_idx], y_outer_train.iloc[inner_val_idx]

                    # Normalize the data
                    scaler = StandardScaler()
                    scaler.fit(X_inner_train)
                    X_inner_train = pd.DataFrame(
                        scaler.transform(X_inner_train),
                        columns=X_inner_train.columns,
                        index=X_inner_train.index
                    )
                    X_inner_val = pd.DataFrame(
                        scaler.transform(X_inner_val),
                        columns=X_inner_val.columns,
                        index=X_inner_val.index
                    )

                    X_inner_train = X_inner_train[selected_features]
                    X_inner_val = X_inner_val[selected_features]

                    # Optuna study for hyperparameter tuning
                    study = optuna.create_study(
                        sampler=TPESampler(seed=round_seed),
                        direction="maximize"
                    )
                    study.optimize(
                        lambda trial: self._objective(
                            trial,
                            X_inner_train,
                            y_inner_train,
                            X_inner_val,
                            y_inner_val
                        ),
                        n_trials=self.n_optuna_trials,
                        show_progress_bar=False,
                        n_jobs=-1
                    )

                    # If current trial score is better than the previous best
                    # trial score, update the best trial and hyperparameters
                    trial_score = study.best_value
                    if trial_score > inner_best_score:
                        inner_best_score = trial_score
                        best_trial = study.best_trial
                        best_hyperparams = best_trial.params

                # Resume Outer Loop --------------------------------------------
                print()
                print(f"Best trial score: {inner_best_score}")
                print(f"Best hyperparameters: {best_hyperparams}")

                best_hyperparams = best_trial.params

                if self.classifier_type == "LinearDiscriminantAnalysis":
                    if best_hyperparams.get("solver") == "svd":
                        # If solver is 'svd', shrinkage is not supported.
                        # Remove the shrinkage parameter from the dictionary.
                        best_hyperparams["shrinkage"] = None

                # Train the model with the best hyperparameters from the inner
                # loop on the outer training set

                # Normalize the outer training set
                scaler = StandardScaler()
                scaler.fit(X_outer_train)
                X_outer_train_scaled = pd.DataFrame(
                    scaler.transform(X_outer_train),
                    columns=X_outer_train.columns,
                    index=X_outer_train.index
                )
                X_outer_test_scaled = pd.DataFrame(
                    scaler.transform(X_outer_test),
                    columns=X_outer_test.columns,
                    index=X_outer_test.index
                )

                X_outer_train_final = X_outer_train_scaled[selected_features]
                X_outer_test_final = X_outer_test_scaled[selected_features]

                model = self.classifier.set_params(**best_hyperparams)
                model.fit(X_outer_train_final, y_outer_train)
                y_outer_pred = model.predict(X_outer_test_final)

                # Calculate metrics
                accuracy = accuracy_score(y_outer_test, y_outer_pred)
                precision = precision_score(y_outer_test, y_outer_pred)
                recall = recall_score(y_outer_test, y_outer_pred)
                specificity = specificity_score(y_outer_test, y_outer_pred)
                mcc = matthews_corrcoef(y_outer_test, y_outer_pred)
                f1 = f1_score(y_outer_test, y_outer_pred)
                roc_auc = roc_auc_score(y_outer_test, y_outer_pred)

                print(f"Outer fold {outer_fold_idx + 1}/{self.n_outer_folds} results:")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"Specificity: {specificity:.4f}")
                print(f"MCC: {mcc:.4f}")
                print(f"F1: {f1:.4f}")
                print(f"ROC AUC: {roc_auc:.4f}")

                self.results.append({
                    "round": round_,
                    "outer_loop": outer_fold_idx,
                    "inner_best_score": inner_best_score,
                    "best_hyperparams": best_hyperparams,
                    "selected_features": selected_features,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "specificity": specificity,
                    "mcc": mcc,
                    "f1": f1,
                    "roc_auc": roc_auc
                })

        # Save the summary of the results --------------------------------------
        self.summary = pd.DataFrame(self.results)
        self.summary["best_hyperparams"] = [d["best_hyperparams"] for d in self.results]
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
        self.summary.to_csv(
            os.path.join(self.results_dir, f"{self.classifier_type}_summary.csv"),
            index=False
        )

        # Train the final model ------------------------------------------------
        best_round = self.summary.loc[self.summary["mean_mcc"].idxmax()]
        # Get the best hyperparams and features
        best_hyperparams = best_round["best_hyperparams"]
        best_features = best_round["selected_features"]
        best_model = VALID_MODELS[self.classifier_type].set_params(**best_hyperparams)
        # Feature selection and normalization
        X = self.X[best_features]
        scaler = StandardScaler()
        scaler.fit(X)
        X = pd.DataFrame(
            scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        # Fit the model
        best_model.fit(X, self.y)

        # Save the model, scaler, and selected features
        self._save_model(best_model, self.classifier_type)
        self._save_scaler(scaler, self.classifier_type)
        self._save_features(best_features, self.classifier_type)


def pipeline(
        classifiers, df, target, models_dir, results_dir,
        n_rounds, n_outer_folds, n_inner_folds,
        n_features_to_select, n_optuna_trials,
        validation_set_fraction, seed
    ):
    # --Find a good holdout set that does not contain any NaN values--
    # This is done to ensure that the holdout set is not affected by
    # the imputation of NaN values in the training set.

    # split to datasets with and without NaN values
    n = df.shape[0]
    nan_mask = df.isna().any(axis=1)
    df_with_nans = df[nan_mask]
    df_without_nans = df[~nan_mask]

    # get a validation (holdout) set from the dataset without NaN values
    val_set_n = int(n * validation_set_fraction)
    val_set = df_without_nans.sample(n=val_set_n, random_state=seed)
    df_without_nans = df_without_nans.drop(val_set.index)

    # recombine the datasets
    df = pd.concat([df_with_nans, df_without_nans], ignore_index=True)

    # --Fill NaN values with the median of the column--
    X = df.drop(columns=[target])
    y = df[target]
    y = y.replace({"M": 1, "B": 0})  # Convert to binary classification
    X = fill_nans_with_median(X, y, True)

    # --Run the nCV for each classifier--
    for clf in classifiers:
        print()
        print(f"--- {clf} ---")
        print()
        ncv = NestedCrossValidation(
            classifier=clf,
            X=X,
            y=y,
            models_dir=models_dir,
            results_dir=results_dir,
            n_rounds=n_rounds,
            n_outer_folds=n_outer_folds,
            n_inner_folds=n_inner_folds,
            n_features_to_select=n_features_to_select,
            n_optuna_trials=n_optuna_trials,
            metric=matthews_corrcoef,
            random_state_base=seed
        )
        ncv.run()

    print("Pipeline completed.")
    # Return the validation set ------------------------------------------------
    return val_set


