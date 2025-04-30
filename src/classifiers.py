import os

import numpy as np

import pandas as pd

from src.preprocessing import fill_nans_with_median

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
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


def time():
    """Returns the current time in a formatted string."""
    now = datetime.now()
    return f"{now:%Y/%m/%d-%H:%M:%S}"

VALID_MODELS = {
    "LogisticRegression":   LogisticRegression(),
    "GaussianNB":  GaussianNB(),
    "LinearDiscriminantAnalysis":  LinearDiscriminantAnalysis(),
    "SVC":  SVC(),
    "RandomForestClassifier":   RandomForestClassifier(),
    "LGBMClassifier": LGBMClassifier()
}

class Classifier:

    def __init__(
            self,
            model_type: str,
            dataset: pd.DataFrame,
            target: str,
            models_dir: str,
            nCV_rounds: int = 10,
            outer_folds: int = 5,
            inner_folds: int = 3,
            num_features: int = 5,
            random_state_base: int = 42,
        ):
        self.simple_log = ""

        # Checks
        if model_type not in VALID_MODELS:
            raise ValueError(f"Invalid model type: {model_type}. "
                             f"Valid options are: {VALID_MODELS}")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            self.simple_log += f"Directory {models_dir} created."

        self.model = VALID_MODELS[model_type]
        self.model_type = self.model.__class__.__name__
        self.X = dataset.drop(columns=[target])
        self.y = dataset[target]
        self.nCV_rounds = nCV_rounds
        self.outer_folds = outer_folds
        self.inner_folds = inner_folds
        self.models_dir = models_dir
        self.num_features = num_features
        self.random_state_base = random_state_base
        self.random_states = [
            self.random_state_base + i
            for i in range(self.nCV_rounds)
        ]

        self.simple_log += f"\n[{time()}] Classifier class created\n\n"
        self.simple_log += "-"* 50 + "\n"
        self.simple_log += f"| Model type: {self.model_type:>34} |\n"
        self.simple_log += f"| Models directory: {self.models_dir:>28} |\n"
        self.simple_log += f"| Dataset shape: {str(self.X.shape):>31} |\n"
        self.simple_log += f"| Target shape: {str(self.y.shape):>32} |\n"
        self.simple_log += f"| Number of CV rounds: {self.nCV_rounds:>25} |\n"
        self.simple_log += f"| Outer folds: {self.outer_folds:>33} |\n"
        self.simple_log += f"| Inner folds: {self.inner_folds:>33} |\n"
        self.simple_log += f"| Random state base: {self.random_state_base:>27} |\n"
        self.simple_log += "-"* 50 + "\n"

        self.hyperparameter_space = self.get_hyperparameter_space(cv_round=0)
        self.simple_log += f"[{time()}] Hyperparameter space created.\n"

    def fit(self, X, y, params=None):
        """Fit the model with optional hyperparameters."""
        self.model = self.create_model(params)
        self.model.fit(X, y)
        self.fitted = True
        self.simple_log += f"[{time()}] Model fitted.\n"
        self.simple_log += "-"* 50 + "\n"

    def predict(self, X):
        """Predict using the trained model."""
        if not hasattr(self, 'fitted') or not self.fitted:
            raise RuntimeError("Model must be fitted before predicting.")
        return self.model.predict(X)

    def save_model(self, filename=None):
        """Save the fitted model to disk using joblib."""
        if not hasattr(self, 'fitted') or not self.fitted:
            raise RuntimeError("Model must be fitted before saving.")
        
        if filename is None:
            filename = f"{self.model_type}_model.joblib"
        
        self.model_path = os.path.join(self.models_dir, filename)
        joblib.dump(self.model, self.model_path)
        self.simple_log += f"[{time()}] Model saved to {self.model_path}.\n"

    def load_model(self, filepath):
        """Load a model from disk using joblib."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model file found at: {filepath}")
        
        self.model = joblib.load(filepath)
        self.fitted = True
        self.model_path = filepath
        self.simple_log += f"[{time()}] Model loaded from {filepath}.\n"

    def get_hyperparameter_space(self, cv_round):
        """Returns the hyperparameter space for the model type."""
        if self.model_type == "LogisticRegression":
            return {
                "penalty": CategoricalDistribution(["elasticnet"]),
                "C": FloatDistribution(1e-5, 1e5, log=True),
                "fit_intercept": CategoricalDistribution([True, False]),
                "solver": CategoricalDistribution(["lbfgs", "liblinear", "sag", "saga"]),
                "random_state": self.random_states[cv_round],
            }
        elif self.model_type == "GaussianNB":
            return {}
        elif self.model_type == "LinearDiscriminantAnalysis":
            return {
                "solver": CategoricalDistribution(["svd", "lsqr", "eigen"]),
                "shrinkage": CategoricalDistribution([None, 'auto'])
            }
        elif self.model_type == "SVC":
            return {
                "C": FloatDistribution(1e-5, 1e5, log=True),
                "kernel": CategoricalDistribution(["linear", "poly", "rbf", "sigmoid"]),
                "degree": IntDistribution(2, 7),
                "gamma": CategoricalDistribution(["scale", "auto"]),
                "random_state": self.random_states[cv_round],
            }
        elif self.model_type == "RandomForestClassifier":
            return {
                "n_estimators": IntDistribution(10, 1000),
                "criterion": CategoricalDistribution(["gini", "entropy", "log_loss"]),
                "max_depth": IntDistribution(10, 50),
                "min_samples_split": IntDistribution(2, 20),
                "min_samples_leaf": IntDistribution(1, 20),
                "max_features": CategoricalDistribution(["sqrt", "log2"]),
                "random_state": self.random_states[cv_round]
            }
        elif self.model_type == "LGBMClassifier":
            return {
                "boosting_type": CategoricalDistribution(["gbdt", "dart"]),
                "num_leaves": IntDistribution(20, 150),
                "learning_rate": FloatDistribution(1e-3, 0.5),
                "n_estimators": IntDistribution(50, 500)
            }
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

    def create_model(self, params=None):
        """Create a model with given parameters (from Optuna)"""
        if params is None or self.model_type == "GaussianNB":
            return VALID_MODELS[self.model_type]
        return VALID_MODELS[self.model_type].__class__(**params)


class NestedCV:
    def __init__(
            self,
            classifier: Classifier,
            rounds: int = 10,
            outer_folds: int = 5,
            inner_folds: int = 3,
            optuna_trials: int = 10
        ):
        self.classifier = classifier
        self.rounds = rounds
        self.outer_folds = outer_folds
        self.inner_folds = inner_folds
        self.optuna_trials = optuna_trials
        self.random_states = self.classifier.random_states

        self.results = []

        self.simple_log = ""
        self.simple_log += f"[{time()}] NestedCV class created.\n"


    def run(self):
        """Runs the nested cross-validation."""
        self.simple_log += f"[{time()}] Running nested cross-validation.\n"
        self.simple_log += "-"* 50 + "\n"

        for cv_round, seed in enumerate(self.random_states):
            self.simple_log += f"[{time()}] Round {cv_round + 1} with seed {seed}.\n"

            # Outer loops - Stratified KFold Cross Validation
            outer_cv = StratifiedKFold(
                n_splits=self.outer_folds,
                shuffle=True,
                random_state=seed
            )

            for outer_fold, (outer_train_index, outer_test_index) in enumerate(
                outer_cv.split(
                    self.classifier.X,
                    self.classifier.y
                )
            ):
                # 1. Extract raw DataFrames
                X_train_df = self.classifier.X.iloc[outer_train_index].copy()
                y_train = self.classifier.y.iloc[outer_train_index]
                X_test_df = self.classifier.X.iloc[outer_test_index].copy()
                y_test = self.classifier.y.iloc[outer_test_index]

                self.simple_log += f"[{time()}] Outer fold {outer_fold + 1}.\n"
                self.simple_log += f"Train shape: {X_train_df.shape}.\n"
                self.simple_log += f"Test shape: {X_test_df.shape}.\n"
                self.simple_log += "-"* 50 + "\n"

                # 2. Apply mRMR feature selection
                desired_num_features = min(self.classifier.num_features, X_train_df.shape[1])
                selected_features = mrmr_classif(X_train_df, y_train, K=desired_num_features)
                X_train_df = X_train_df[selected_features]
                X_test_df = X_test_df[selected_features]
                self.simple_log += f"[{time()}] mRMR selected {len(selected_features)} features.\n"

                # 3. Preprocess (scaling)
                scaler = StandardScaler()
                scaler.fit(X_train_df)
                X_train = scaler.transform(X_train_df)
                X_test = scaler.transform(X_test_df)
                self.simple_log += f"[{time()}] Data scaled after mRMR.\n"
                self.simple_log += "-"* 50 + "\n"

                # 4. Hyperparameter tuning with Optuna
                study = optuna.create_study(
                    sampler=TPESampler(seed=seed),
                    direction="maximize"
                )
                study.optimize(
                    lambda trial: self._objective(trial, X_train, y_train, cv_round),
                    n_trials=self.optuna_trials,
                    n_jobs=-1
                )

                best_params = study.best_params
                best_model = self.classifier.create_model(best_params)
                best_model.fit(X_train, y_train)

                y_pred = best_model.predict(X_test)

                metrics_result = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, zero_division=0),
                    "recall": recall_score(y_test, y_pred, zero_division=0),
                    "f1": f1_score(y_test, y_pred)
                }

                self.results.append({
                    "round": cv_round,
                    "outer_fold": outer_fold,
                    "metrics": metrics_result,
                    "best_params": best_params,
                    "selected_features": selected_features
                })

                self.simple_log += f"[{time()}] Outer Fold {outer_fold + 1} completed.\n"
                self.simple_log += f"Accuracy: {metrics_result['accuracy']:.4f}.\n"
                self.simple_log += f"F1: {metrics_result['f1']:.4f}\n"
                self.simple_log += "-" * 50 + "\n"

        return self.results


    def _objective(self, trial, X, y, cv_round):
        """Objective function for Optuna that handles conditional hyperparameters."""
        params = {}
        
        # Create appropriate hyperparameter suggestions based on model type
        if self.classifier.model_type == "LogisticRegression":
            penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet", "none"])
            params["penalty"] = penalty
            
            # Conditional solver based on penalty
            if penalty == "elasticnet":
                params["solver"] = "saga"  # Only saga supports elasticnet
                params["l1_ratio"] = trial.suggest_float("l1_ratio", 0, 1)
            elif penalty == "l1":
                params["solver"] = trial.suggest_categorical("solver", ["liblinear", "saga"])
            elif penalty == "l2":
                params["solver"] = trial.suggest_categorical("solver", ["lbfgs", "liblinear", "sag", "saga"])
            else:  # "none"
                params["solver"] = trial.suggest_categorical("solver", ["lbfgs", "sag", "saga"])
                
            params["C"] = trial.suggest_float("C", 1e-5, 1e5, log=True)
            params["fit_intercept"] = trial.suggest_categorical("fit_intercept", [True, False])
            params["max_iter"] = trial.suggest_int("max_iter", 100, 1000)
            params["random_state"] = self.classifier.random_states[cv_round]
        
        elif self.classifier.model_type == "GaussianNB":
            # GaussianNB has no hyperparameters to tune
            pass
        
        elif self.classifier.model_type == "LinearDiscriminantAnalysis":
            solver = trial.suggest_categorical("solver", ["svd", "lsqr", "eigen"])
            params["solver"] = solver
            
            # shrinkage is only relevant for 'lsqr' and 'eigen' solvers
            if solver in ["lsqr", "eigen"]:
                shrinkage_type = trial.suggest_categorical("shrinkage_type", ["auto", "manual", "none"])
                if shrinkage_type == "auto":
                    params["shrinkage"] = "auto"
                elif shrinkage_type == "manual":
                    params["shrinkage"] = trial.suggest_float("shrinkage_value", 0, 1)
                # If 'none', don't set shrinkage
            
            # n_components is meaningful only when n_classes > 1
            # In binary classification, max is 1
            # Assuming we might have multi-class, we'll add this
            if len(np.unique(y)) > 2:  # More than binary classification
                n_components_max = min(len(np.unique(y)) - 1, X.shape[1])
                if n_components_max > 1:  # Only suggest if we have room for multiple components
                    params["n_components"] = trial.suggest_int("n_components", 1, n_components_max)
        
        elif self.classifier.model_type == "SVC":
            kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
            params["kernel"] = kernel
            
            # Only suggest degree for poly kernel
            if kernel == "poly":
                params["degree"] = trial.suggest_int("degree", 2, 7)
            
            # Gamma is only relevant for 'poly', 'rbf', and 'sigmoid'
            if kernel in ["poly", "rbf", "sigmoid"]:
                params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
            
            params["C"] = trial.suggest_float("C", 1e-5, 1e5, log=True)
            params["probability"] = True  # Enable probability estimates
            params["random_state"] = self.classifier.random_states[cv_round]
        
        elif self.classifier.model_type == "RandomForestClassifier":
            params["n_estimators"] = trial.suggest_int("n_estimators", 10, 1000)
            params["criterion"] = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])
            
            # Control tree depth
            max_depth_enabled = trial.suggest_categorical("max_depth_enabled", [True, False])
            if max_depth_enabled:
                params["max_depth"] = trial.suggest_int("max_depth", 10, 50)
            else:
                params["max_depth"] = None
                
            params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 20)
            params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 20)
            
            # Feature selection strategy
            params["max_features"] = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
            
            # Bootstrap options
            params["bootstrap"] = trial.suggest_categorical("bootstrap", [True, False])
            
            # Class weight options
            class_weight_option = trial.suggest_categorical("class_weight_option", ["balanced", "balanced_subsample", None])
            if class_weight_option is not None:
                params["class_weight"] = class_weight_option
                
            params["random_state"] = self.classifier.random_states[cv_round]
        
        elif self.classifier.model_type == "LGBMClassifier":
            params["boosting_type"] = trial.suggest_categorical("boosting_type", ["gbdt", "dart", "goss"])
            
            # Handle incompatible boosting types
            if params["boosting_type"] == "goss":
                params["subsample"] = 1.0  # Cannot use bagging with GOSS
            else:
                params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
                
            # Common parameters
            params["num_leaves"] = trial.suggest_int("num_leaves", 20, 150)
            params["learning_rate"] = trial.suggest_float("learning_rate", 1e-3, 0.5, log=True)
            params["n_estimators"] = trial.suggest_int("n_estimators", 50, 500)
            params["min_child_samples"] = trial.suggest_int("min_child_samples", 5, 100)
            
            # Regularization
            params["reg_alpha"] = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True)
            params["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)
            
            # Feature fraction for better generalization
            params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 1.0)
            
            # Class weight
            params["class_weight"] = trial.suggest_categorical("class_weight", ["balanced", None])
            
            params["random_state"] = self.classifier.random_states[cv_round]
        
        else:
            raise ValueError(f"Invalid model type: {self.classifier.model_type}")
        
        # Create and evaluate the model
        model = self.classifier.create_model(params)
        
        inner_cv = StratifiedKFold(
            n_splits=self.inner_folds, 
            shuffle=True, 
            random_state=self.classifier.random_states[cv_round]
        )
        
        # Use F1 score as the optimization metric
        score = cross_val_score(
            model, X, y, 
            cv=inner_cv, 
            scoring="f1", 
            n_jobs=-1  # Use all available cores
        ).mean()
        
        return score


def pipeline(
        dataset: pd.DataFrame,
        models_dir: str
    ):
    """Complete pipeline helper function to automate everything in the assignment.

    Parameters
    ----------

    Returns
    -------
    """
    for model in VALID_MODELS.keys():
        clf = Classifier(model, dataset, "diagnosis", models_dir=models_dir)
        ncv = NestedCV(clf)
        results = ncv.run()
        print(model)
        print(results)


if __name__ == "__main__":
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    df = pd.read_csv(os.path.join(data_dir, "breast_cancer.csv"))
    pipeline(dataset=df, models_dir=models_dir)