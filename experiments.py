import json
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from grader_contracts.linear_models import ClassificationInput, RegressionInput
from linear_models import (
    RANDOM_STATE,
    classification_metrics,
    compare_classical_models,
    train_linear_regression,
    train_logistic_classifier,
)

TEST_SIZE = 0.2
REGRESSION_TARGET = "Performance Index"
CLASSIFICATION_TARGET = "Creditability"
RESULTS_PATH = Path("experiment_results.json")
PLOTS_DIR = Path("plots")


def load_student_performance(path: str = "Student_Performance.csv") -> tuple[pd.DataFrame, pd.Series]:
    students = pd.read_csv(path)
    students["Extracurricular Activities"] = students["Extracurricular Activities"].map({"Yes": 1, "No": 0})
    return students.drop(columns=REGRESSION_TARGET), students[REGRESSION_TARGET]


def load_german_credit(path: str = "german.csv") -> tuple[pd.DataFrame, pd.Series]:
    credits = pd.read_csv(path, sep=";")
    return credits.drop(columns=CLASSIFICATION_TARGET), credits[CLASSIFICATION_TARGET]


def regression_input() -> RegressionInput:
    features, target = load_student_performance()
    train_x, test_x, train_y, test_y = train_test_split(
        features, target, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return RegressionInput(train_x, train_y, test_x, test_y)


def classification_input() -> ClassificationInput:
    features, target = load_german_credit()
    train_x, test_x, train_y, test_y = train_test_split(
        features, target, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=target
    )
    return ClassificationInput(train_x, train_y, test_x, test_y)


def run_experiments() -> dict:
    regression_data = regression_input()
    regression = train_linear_regression(regression_data)
    test_mse = float(np.mean((regression.predictions - regression_data.test_target.to_numpy()) ** 2))

    classification_data = classification_input()
    own = train_logistic_classifier(classification_data)
    baselines = compare_classical_models(classification_data)

    _plot_loss(regression.loss_history, "MSE", "Линейная регрессия: MSE на train", "loss_linear_regression.png")
    _plot_loss(own.loss_history, "BCE", "Логистическая регрессия: BCE на train", "loss_logistic_regression.png")

    return {
        "linear_regression": {
            "test_mse": test_mse,
            "train_mse_first_epoch": regression.loss_history[0],
            "train_mse_last_epoch": regression.loss_history[-1],
            "weights": dict(zip(regression_data.train_features.columns, map(float, regression.weights))),
            "bias": regression.bias,
        },
        "classification": {
            "own_logistic": asdict(classification_metrics(classification_data.test_target, own.probabilities)),
            "logistic_regression": asdict(baselines.logistic_regression),
            "decision_tree": asdict(baselines.decision_tree),
            "knn": asdict(baselines.knn),
            "own_logistic_bce_first_epoch": own.loss_history[0],
            "own_logistic_bce_last_epoch": own.loss_history[-1],
        },
        "hyperparameters": hyperparameter_sweeps(regression_data, classification_data),
    }


def hyperparameter_sweeps(regression_data: RegressionInput, classification_data: ClassificationInput) -> dict:
    test_target = regression_data.test_target.to_numpy()
    regression_learning_rates = {}
    for learning_rate in (0.001, 0.01, 0.05, 0.2):
        result = train_linear_regression(replace(regression_data, learning_rate=learning_rate))
        regression_learning_rates[str(learning_rate)] = {
            "train_mse_last_epoch": result.loss_history[-1],
            "test_mse": float(np.mean((result.predictions - test_target) ** 2)),
        }

    logistic_learning_rates = {}
    for learning_rate in (0.01, 0.1, 0.5):
        result = train_logistic_classifier(replace(classification_data, learning_rate=learning_rate))
        metrics = classification_metrics(classification_data.test_target, result.probabilities)
        logistic_learning_rates[str(learning_rate)] = {
            "bce_last_epoch": result.loss_history[-1],
            "roc_auc": metrics.roc_auc,
            "f1": metrics.f1,
        }

    own = train_logistic_classifier(classification_data)
    thresholds = {
        str(threshold): asdict(classification_metrics(classification_data.test_target, own.probabilities, threshold))
        for threshold in (0.3, 0.5, 0.7)
    }

    train_x, test_x = classification_data.train_features, classification_data.test_features
    train_y, test_y = classification_data.train_target, classification_data.test_target
    tree_depths = {}
    for depth in (3, 5, 10, None):
        tree = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE).fit(train_x, train_y)
        tree_depths[str(depth)] = classification_metrics(test_y, tree.predict_proba(test_x)[:, 1]).roc_auc
    knn_neighbors = {}
    for neighbors in (5, 15, 31):
        knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=neighbors)).fit(train_x, train_y)
        knn_neighbors[str(neighbors)] = classification_metrics(test_y, knn.predict_proba(test_x)[:, 1]).roc_auc

    return {
        "linear_regression_learning_rate": regression_learning_rates,
        "own_logistic_learning_rate": logistic_learning_rates,
        "own_logistic_threshold": thresholds,
        "decision_tree_max_depth_roc_auc": tree_depths,
        "knn_neighbors_roc_auc": knn_neighbors,
    }


def _plot_loss(history: list[float], label: str, title: str, filename: str) -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(range(1, len(history) + 1), history)
    axis.set_xlabel("Эпоха")
    axis.set_ylabel(label)
    if max(history) / min(history) > 10:
        axis.set_yscale("log")
    axis.set_title(title)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / filename, dpi=120)
    plt.close(figure)


def main() -> None:
    results = run_experiments()
    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
