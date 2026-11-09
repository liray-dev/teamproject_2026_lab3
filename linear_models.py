from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)

RANDOM_STATE = 42
DECISION_THRESHOLD = 0.5
PROBABILITY_EPSILON = 1e-12
TREE_MAX_DEPTH = 5
KNN_NEIGHBORS = 15


@dataclass(frozen=True)
class _Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> _Standardizer:
        scale = features.std(axis=0)
        return cls(mean=features.mean(axis=0), scale=np.where(scale > 0, scale, 1.0))

    def transform(self, features: np.ndarray) -> np.ndarray:
        return (features - self.mean) / self.scale

    def unscale(self, weights: np.ndarray, bias: float) -> tuple[np.ndarray, float]:
        original_weights = weights / self.scale
        return original_weights, float(bias - original_weights @ self.mean)


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучите собственную линейную регрессию на train и верните прогноз для test."""
    train_features, test_features = _as_matrix(data.train_features), _as_matrix(data.test_features)
    target = _as_vector(data.train_target)
    standardizer = _Standardizer.fit(train_features)
    scaled = standardizer.transform(train_features)

    weights, bias = np.zeros(scaled.shape[1]), 0.0
    loss_history = []
    for _ in range(data.epochs):
        error = scaled @ weights + bias - target
        loss_history.append(float(np.mean(error**2)))
        weights -= data.learning_rate * 2 * scaled.T @ error / len(target)
        bias -= data.learning_rate * 2 * float(error.mean())

    weights, bias = standardizer.unscale(weights, bias)
    return RegressionResult(
        weights=weights,
        bias=bias,
        loss_history=loss_history,
        predictions=test_features @ weights + bias,
    )


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучите собственный логистический классификатор на train и верните прогноз для test."""
    train_features, test_features = _as_matrix(data.train_features), _as_matrix(data.test_features)
    target = _as_binary_vector(data.train_target)
    standardizer = _Standardizer.fit(train_features)
    scaled = standardizer.transform(train_features)

    weights, bias = np.zeros(scaled.shape[1]), 0.0
    loss_history = []
    for _ in range(data.epochs):
        probabilities = _sigmoid(scaled @ weights + bias)
        loss_history.append(binary_cross_entropy(target, probabilities))
        error = probabilities - target
        weights -= data.learning_rate * scaled.T @ error / len(target)
        bias -= data.learning_rate * float(error.mean())

    weights, bias = standardizer.unscale(weights, bias)
    test_probabilities = _sigmoid(test_features @ weights + bias)
    return ClassificationResult(
        weights=weights,
        bias=bias,
        loss_history=loss_history,
        probabilities=test_probabilities,
        predictions=(test_probabilities >= DECISION_THRESHOLD).astype(int),
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравните готовые Logistic Regression, Decision Tree и KNN на одном split.

    После запуска запишите реальные результаты в experiment_results.json:
    {"linear_regression":{"test_mse": число},
     "classification":{"own_logistic": {"roc_auc": ..., "accuracy": ..., "precision": ..., "recall": ..., "f1": ...},
                       "logistic_regression": {...}, "decision_tree": {...}, "knn": {...}}}.
    """
    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=data.epochs, random_state=RANDOM_STATE)
        ),
        "decision_tree": DecisionTreeClassifier(max_depth=TREE_MAX_DEPTH, random_state=RANDOM_STATE),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS)),
    }
    train_features, test_features = _as_matrix(data.train_features), _as_matrix(data.test_features)
    train_target, test_target = _as_binary_vector(data.train_target), _as_binary_vector(data.test_target)
    metrics = {}
    for name, model in models.items():
        model.fit(train_features, train_target)
        probabilities = model.predict_proba(test_features)[:, 1]
        metrics[name] = classification_metrics(test_target, probabilities)
    return BaselineComparisonResult(**metrics)


def classification_metrics(target, probabilities, threshold: float = DECISION_THRESHOLD) -> ClassificationMetrics:
    target = _as_binary_vector(target)
    probabilities = _as_vector(probabilities)
    predictions = (probabilities >= threshold).astype(int)
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(target, probabilities)),
        accuracy=float(accuracy_score(target, predictions)),
        precision=float(precision_score(target, predictions, zero_division=0)),
        recall=float(recall_score(target, predictions, zero_division=0)),
        f1=float(f1_score(target, predictions, zero_division=0)),
    )


def binary_cross_entropy(target: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)
    return float(-np.mean(target * np.log(clipped) + (1 - target) * np.log(1 - clipped)))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 0.5 * (1 + np.tanh(0.5 * values))


def _as_matrix(values) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    return matrix.reshape(-1, 1) if matrix.ndim == 1 else matrix


def _as_vector(values) -> np.ndarray:
    return np.asarray(values, dtype=float).ravel()


def _as_binary_vector(values) -> np.ndarray:
    vector = _as_vector(values)
    if not np.isin(vector, (0.0, 1.0)).all():
        raise ValueError("binary target must contain only 0 and 1")
    return vector
