import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "03_data_processing" / "03_minibatch_training"
spec = importlib.util.spec_from_file_location("minibatch_solution", TOPIC / "reference" / "solution.py")
assert spec is not None and spec.loader is not None
solution = importlib.util.module_from_spec(spec); spec.loader.exec_module(solution)


def flatten(epoch: tuple[np.ndarray, ...]) -> np.ndarray:
    return np.concatenate(epoch)


def schedules_equal(first, second) -> bool:
    return all(
        len(epoch_a) == len(epoch_b)
        and all(np.array_equal(a, b) for a, b in zip(epoch_a, epoch_b))
        for epoch_a, epoch_b in zip(first, second)
    )


def test_no_shuffle_batches_and_tail_are_exact() -> None:
    schedule = solution.batch_index_schedule(7, 3, 2, shuffle=False)
    assert [[batch.tolist() for batch in epoch] for epoch in schedule] == [
        [[0, 1, 2], [3, 4, 5], [6]],
        [[0, 1, 2], [3, 4, 5], [6]],
    ]


def test_every_shuffled_epoch_covers_each_sample_once() -> None:
    schedule = solution.batch_index_schedule(17, 4, 5, random_state=9)
    for epoch in schedule:
        np.testing.assert_array_equal(np.sort(flatten(epoch)), np.arange(17))
        assert [len(batch) for batch in epoch] == [4, 4, 4, 4, 1]


def test_same_seed_reproduces_complete_schedule() -> None:
    first = solution.batch_index_schedule(20, 6, 4, random_state=12)
    second = solution.batch_index_schedule(20, 6, 4, random_state=12)
    other = solution.batch_index_schedule(20, 6, 4, random_state=13)
    assert schedules_equal(first, second)
    assert not schedules_equal(first, other)


def test_rng_is_not_reset_to_same_order_each_epoch() -> None:
    schedule = solution.batch_index_schedule(20, 20, 3, random_state=4)
    assert not np.array_equal(schedule[0][0], schedule[1][0])
    assert not np.array_equal(schedule[1][0], schedule[2][0])


def test_drop_last_and_oversized_batch_boundaries() -> None:
    dropped = solution.batch_index_schedule(7, 3, 1, shuffle=False, drop_last=True)
    assert [batch.tolist() for batch in dropped[0]] == [[0, 1, 2], [3, 4, 5]]
    kept = solution.batch_index_schedule(3, 10, 1, shuffle=False)
    assert kept[0][0].tolist() == [0, 1, 2]
    with pytest.raises(ValueError, match="全部"):
        solution.batch_index_schedule(3, 10, 1, drop_last=True)


def test_take_minibatch_keeps_features_and_targets_synchronized() -> None:
    X = np.column_stack((np.arange(6), np.arange(6) * 10)).astype(float)
    y = np.arange(6).astype(float) + 100; X_before = X.copy(); y_before = y.copy()
    indices = np.array([4, 1, 5])
    X_batch, y_batch = solution.take_minibatch(X, y, indices)
    np.testing.assert_array_equal(X_batch[:, 0], y_batch - 100)
    X_batch[0, 0] = -999; y_batch[0] = -999
    np.testing.assert_array_equal(X, X_before); np.testing.assert_array_equal(y, y_before)


@pytest.mark.parametrize(
    "indices",
    [np.array([], dtype=int), np.array([0, 0]), np.array([-1, 1]), np.array([0, 5]), np.array([0., 1.])],
)
def test_invalid_batch_indices_are_rejected(indices: np.ndarray) -> None:
    with pytest.raises(ValueError):
        solution.take_minibatch(np.ones((5, 2)), np.ones(5), indices)


def test_linear_gradient_matches_hand_formula() -> None:
    X = np.array([[1., 2.], [3., 4.]])
    y = np.array([1., 2.]); weights = np.array([0.5, -0.25]); bias = 0.1
    residual = X @ weights + bias - y
    loss, gradient, gradient_bias = solution.linear_mse_loss_gradient(X, y, weights, bias)
    assert loss == pytest.approx(0.5 * np.mean(residual ** 2))
    np.testing.assert_allclose(gradient, X.T @ residual / 2)
    assert gradient_bias == pytest.approx(np.mean(residual))


def test_linear_gradient_matches_finite_difference() -> None:
    X = np.array([[1., -1.], [2., 0.5], [-1., 3.]])
    y = np.array([2., -1., 0.5]); weights = np.array([0.2, -0.4]); bias = 0.3
    _, gradient, gradient_bias = solution.linear_mse_loss_gradient(X, y, weights, bias)
    epsilon = 1e-6; numerical = np.empty_like(weights)
    for index in range(len(weights)):
        plus = weights.copy(); minus = weights.copy(); plus[index] += epsilon; minus[index] -= epsilon
        numerical[index] = (
            solution.linear_mse_loss_gradient(X, y, plus, bias)[0]
            - solution.linear_mse_loss_gradient(X, y, minus, bias)[0]
        ) / (2 * epsilon)
    numerical_bias = (
        solution.linear_mse_loss_gradient(X, y, weights, bias + epsilon)[0]
        - solution.linear_mse_loss_gradient(X, y, weights, bias - epsilon)[0]
    ) / (2 * epsilon)
    np.testing.assert_allclose(gradient, numerical, atol=1e-7)
    assert gradient_bias == pytest.approx(numerical_bias, abs=1e-7)


def test_one_full_batch_epoch_equals_one_gradient_step() -> None:
    X = np.array([[1., 2.], [3., 4.], [-1., 0.]])
    y = np.array([1., 2., -1.]); learning_rate = 0.1
    _, gradient, gradient_bias = solution.linear_mse_loss_gradient(X, y, np.zeros(2), 0.0)
    model = solution.fit_minibatch_linear_regression(
        X, y, batch_size=len(X), n_epochs=1, learning_rate=learning_rate, shuffle=False
    )
    np.testing.assert_allclose(model["weights"], -learning_rate * gradient)
    assert model["bias"] == pytest.approx(-learning_rate * gradient_bias)


def test_minibatch_training_reduces_loss_and_recovers_linear_rule() -> None:
    rng = np.random.default_rng(2); X = rng.normal(size=(80, 2)); y = 1.0 + 2.0 * X[:, 0] - 3.0 * X[:, 1]
    model = solution.fit_minibatch_linear_regression(
        X, y, batch_size=8, n_epochs=120, learning_rate=0.05, random_state=6
    )
    assert model["loss_history"].shape == (121,)
    assert model["loss_history"][-1] < 1e-8
    np.testing.assert_allclose(model["weights"], [2., -3.], atol=2e-4)
    assert model["bias"] == pytest.approx(1.0, abs=2e-4)


def test_complete_training_is_reproducible() -> None:
    X = np.arange(30, dtype=float).reshape(10, 3) / 10; y = X @ np.array([1., -2., 0.5]) + 0.2
    first = solution.fit_minibatch_linear_regression(X, y, batch_size=4, n_epochs=10, random_state=8)
    second = solution.fit_minibatch_linear_regression(X, y, batch_size=4, n_epochs=10, random_state=8)
    np.testing.assert_array_equal(first["weights"], second["weights"])
    np.testing.assert_array_equal(first["loss_history"], second["loss_history"])
    assert schedules_equal(first["batch_schedule"], second["batch_schedule"])


def test_invalid_schedule_and_learning_arguments_are_rejected() -> None:
    with pytest.raises(ValueError): solution.batch_index_schedule(0, 2, 1)
    with pytest.raises(ValueError): solution.batch_index_schedule(4, 0, 1)
    with pytest.raises(ValueError): solution.batch_index_schedule(4, 2, 0)
    with pytest.raises(ValueError): solution.batch_index_schedule(4, 2, 1, random_state=True)
    with pytest.raises(ValueError):
        solution.fit_minibatch_linear_regression(np.ones((3, 1)), np.ones(3), batch_size=1, learning_rate=0)


def test_guided_demo_runs_and_shows_alignment() -> None:
    completed = subprocess.run(
        [sys.executable, str(TOPIC / "guided_demo.py")], cwd=ROOT,
        check=True, capture_output=True, text=True, encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    assert "覆盖: [0, 1, 2, 3, 4, 5, 6]" in completed.stdout
    assert "X/y同步: True" in completed.stdout


def test_starter_remains_for_student() -> None:
    starter_spec = importlib.util.spec_from_file_location("minibatch_starter", TOPIC / "starter.py")
    assert starter_spec is not None and starter_spec.loader is not None
    starter = importlib.util.module_from_spec(starter_spec); starter_spec.loader.exec_module(starter)
    with pytest.raises(NotImplementedError):
        starter.batch_index_schedule(5, 2, 1)
