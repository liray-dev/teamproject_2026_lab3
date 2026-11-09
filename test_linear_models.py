import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from experiments import classification_input, regression_input, run_experiments
from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    RegressionInput,
)
from linear_models import (
    binary_cross_entropy,
    classification_metrics,
    compare_classical_models,
    train_linear_regression,
    train_logistic_classifier,
)


@pytest.fixture
def linear_data():
    generator = np.random.default_rng(0)
    features = generator.normal(size=(300, 3)) * [1.0, 50.0, 0.01] + [0.0, 200.0, 5.0]
    target = features @ np.array([2.0, -0.3, 400.0]) + 7.0 + generator.normal(scale=0.1, size=300)
    return RegressionInput(features[:240], target[:240], features[240:], target[240:])


@pytest.fixture
def separable_data():
    generator = np.random.default_rng(1)
    features = generator.normal(size=(400, 2)) * [1.0, 100.0]
    target = (features[:, 0] + features[:, 1] / 100.0 > 0).astype(int)
    return ClassificationInput(features[:300], target[:300], features[300:], target[300:])


def test_linear_regression_recovers_coefficients(linear_data):
    result = train_linear_regression(linear_data)
    assert result.weights == pytest.approx([2.0, -0.3, 400.0], rel=0.05)
    assert result.bias == pytest.approx(7.0, abs=1.0)
    assert isinstance(result.bias, float)


def test_linear_regression_shapes_and_history(linear_data):
    result = train_linear_regression(linear_data)
    assert len(result.loss_history) == linear_data.epochs
    assert all(isinstance(value, float) for value in result.loss_history)
    assert np.all(np.diff(result.loss_history) <= 1e-9)
    assert np.shape(result.predictions) == (60,)


def test_linear_regression_predictions_use_returned_weights(linear_data):
    result = train_linear_regression(linear_data)
    assert result.predictions == pytest.approx(linear_data.test_features @ result.weights + result.bias)


def test_linear_regression_matches_ols_on_students():
    data = regression_input()
    result = train_linear_regression(data)
    reference = LinearRegression().fit(data.train_features, data.train_target)
    assert result.weights == pytest.approx(reference.coef_, rel=1e-6)
    assert result.bias == pytest.approx(reference.intercept_, rel=1e-6)


def test_logistic_classifier_separates_classes(separable_data):
    result = train_logistic_classifier(separable_data)
    assert np.mean(result.predictions == separable_data.test_target) >= 0.95
    assert len(result.loss_history) == separable_data.epochs
    assert result.loss_history[0] == pytest.approx(np.log(2))
    assert result.loss_history[-1] < result.loss_history[0]
    assert np.all((result.probabilities >= 0) & (result.probabilities <= 1))
    assert set(np.unique(result.predictions)) <= {0, 1}


def test_logistic_predictions_follow_probabilities(separable_data):
    result = train_logistic_classifier(separable_data)
    np.testing.assert_array_equal(result.predictions, (result.probabilities >= 0.5).astype(int))
    expected = 1 / (1 + np.exp(-(separable_data.test_features @ result.weights + result.bias)))
    assert result.probabilities == pytest.approx(expected)


def test_logistic_classifier_rejects_non_binary_target(separable_data):
    data = ClassificationInput(
        separable_data.train_features, separable_data.train_target + 1,
        separable_data.test_features, separable_data.test_target,
    )
    with pytest.raises(ValueError):
        train_logistic_classifier(data)


def test_logistic_close_to_sklearn_on_german():
    data = classification_input()
    own = train_logistic_classifier(data)
    reference = LogisticRegression(C=np.inf, max_iter=5000).fit(
        (data.train_features - data.train_features.mean()) / data.train_features.std(ddof=0), data.train_target
    )
    reference_probabilities = reference.predict_proba(
        (data.test_features - data.train_features.mean()) / data.train_features.std(ddof=0)
    )[:, 1]
    assert np.corrcoef(own.probabilities, reference_probabilities)[0, 1] > 0.99


def test_binary_cross_entropy_is_finite_on_extremes():
    target = np.array([0.0, 1.0])
    assert binary_cross_entropy(target, np.array([0.0, 1.0])) == pytest.approx(0.0, abs=1e-9)
    assert np.isfinite(binary_cross_entropy(target, np.array([1.0, 0.0])))


def test_classification_metrics_simple_case():
    metrics = classification_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9])
    assert metrics == ClassificationMetrics(roc_auc=0.75, accuracy=0.5, precision=0.5, recall=0.5, f1=0.5)


def test_compare_classical_models(separable_data):
    result = compare_classical_models(separable_data)
    assert isinstance(result, BaselineComparisonResult)
    for metrics in (result.logistic_regression, result.decision_tree, result.knn):
        assert isinstance(metrics, ClassificationMetrics)
        assert metrics.accuracy >= 0.85
        assert 0.0 <= metrics.roc_auc <= 1.0


def test_experiment_results_file_is_reproducible():
    saved = json.loads(Path("experiment_results.json").read_text(encoding="utf-8"))
    fresh = run_experiments()
    assert saved["linear_regression"]["test_mse"] == pytest.approx(fresh["linear_regression"]["test_mse"], rel=1e-6)
    for model in ("own_logistic", "logistic_regression", "decision_tree", "knn"):
        assert saved["classification"][model] == pytest.approx(fresh["classification"][model], rel=1e-6)
