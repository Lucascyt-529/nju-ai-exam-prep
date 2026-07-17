import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "watermelon_book" / "05_neural_networks" / "07_som"
SOLUTION = TOPIC / "reference" / "solution.py"


def load_solution_module():
    spec = importlib.util.spec_from_file_location("som_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def clustered_data() -> np.ndarray:
    return np.array([[-2.2], [-2.0], [-1.8], [1.8], [2.0], [2.2]])


def test_bmu_uses_squared_distance_and_earliest_tie() -> None:
    prototypes = np.array([[-1.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    assert solution.best_matching_unit(np.array([2.8, 0.0]), prototypes) == 2
    assert solution.best_matching_unit(np.array([0.0, 0.0]), prototypes) == 0


def test_gaussian_neighborhood_is_symmetric_around_winner() -> None:
    neighborhood = solution.gaussian_neighborhood(2, 5, radius=1.0)
    assert neighborhood.shape == (5,)
    assert neighborhood[2] == pytest.approx(1.0)
    assert neighborhood[1] == pytest.approx(neighborhood[3])
    assert neighborhood[0] == pytest.approx(neighborhood[4])
    assert neighborhood[2] > neighborhood[1] > neighborhood[0]


def test_single_update_matches_formula_and_is_not_in_place() -> None:
    prototypes = np.array([[0.0], [2.0], [4.0]])
    original = prototypes.copy()
    sample = np.array([2.0])
    neighborhood = solution.gaussian_neighborhood(1, 3, radius=1.0)
    updated = solution.update_prototypes(
        prototypes, sample, 1, learning_rate=0.5, radius=1.0
    )
    expected = original + 0.5 * neighborhood[:, None] * (sample[None, :] - original)
    np.testing.assert_allclose(updated, expected)
    np.testing.assert_array_equal(prototypes, original)


def test_decay_has_exact_endpoints_and_is_monotone() -> None:
    values = [solution.exponential_decay(1.0, 0.1, step, 5) for step in range(5)]
    assert values[0] == pytest.approx(1.0)
    assert values[-1] == pytest.approx(0.1)
    assert np.all(np.diff(values) < 0)
    assert solution.exponential_decay(0.5, 0.1, 0, 1) == pytest.approx(0.5)


def test_initialization_is_reproducible_and_inside_feature_ranges() -> None:
    X = np.array([[-2.0, 10.0], [2.0, 20.0]])
    first = solution.initialize_prototypes(X, 4, seed=5)
    second = solution.initialize_prototypes(X, 4, seed=5)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (4, 2)
    assert np.all((first[:, 0] >= -2.0) & (first[:, 0] <= 2.0))
    assert np.all((first[:, 1] >= 10.0) & (first[:, 1] <= 20.0))


def test_quantization_error_matches_hand_calculation() -> None:
    X = np.array([[0.0], [2.0], [5.0]])
    prototypes = np.array([[0.0], [4.0]])
    assert solution.quantization_error(X, prototypes) == pytest.approx((0.0 + 2.0 + 1.0) / 3.0)


def test_training_improves_quantization_and_records_each_epoch() -> None:
    X = clustered_data()
    prototypes, history = solution.train_som(X, n_neurons=4, epochs=40, seed=3)
    assert prototypes.shape == (4, 1)
    assert len(history) == 41
    assert history[-1] < history[0] * 0.3


def test_zero_epochs_returns_initial_prototypes_and_one_error() -> None:
    X = clustered_data()
    prototypes, history = solution.train_som(X, n_neurons=3, epochs=0, seed=7)
    expected = solution.initialize_prototypes(X, 3, seed=7)
    np.testing.assert_array_equal(prototypes, expected)
    assert history == [solution.quantization_error(X, expected)]


def test_training_and_mapping_are_deterministic() -> None:
    X = clustered_data()
    first_prototypes, first_history = solution.train_som(X, n_neurons=4, epochs=20, seed=2)
    second_prototypes, second_history = solution.train_som(X, n_neurons=4, epochs=20, seed=2)
    np.testing.assert_array_equal(first_prototypes, second_prototypes)
    assert first_history == second_history
    np.testing.assert_array_equal(
        solution.map_samples(X, first_prototypes), solution.map_samples(X, second_prototypes)
    )


def test_distant_clusters_map_to_different_map_regions() -> None:
    X = clustered_data()
    prototypes, _ = solution.train_som(X, n_neurons=4, epochs=40, seed=3)
    winners = solution.map_samples(X, prototypes)
    assert winners.shape == (6,)
    assert set(winners[:3]).isdisjoint(set(winners[3:]))


def test_training_does_not_modify_X() -> None:
    X = clustered_data()
    original = X.copy()
    solution.train_som(X, n_neurons=4, epochs=5, seed=1)
    np.testing.assert_array_equal(X, original)


@pytest.mark.parametrize(
    "call",
    [
        lambda: solution.best_matching_unit(np.array([[0.0]]), np.ones((2, 1))),
        lambda: solution.best_matching_unit(np.array([0.0, 1.0]), np.ones((2, 1))),
        lambda: solution.gaussian_neighborhood(-1, 3, 1.0),
        lambda: solution.gaussian_neighborhood(0, 0, 1.0),
        lambda: solution.gaussian_neighborhood(0, 3, 0.0),
        lambda: solution.update_prototypes(
            np.ones((2, 1)), np.array([0.0]), 0, learning_rate=1.1, radius=1.0
        ),
        lambda: solution.exponential_decay(1.0, 0.1, 5, 5),
        lambda: solution.exponential_decay(0.1, 1.0, 0, 5),
        lambda: solution.train_som(np.ones((2, 1)), n_neurons=0, epochs=1),
        lambda: solution.train_som(np.ones((2, 1)), n_neurons=2, epochs=-1),
        lambda: solution.train_som(
            np.ones((2, 1)), n_neurons=2, epochs=1, initial_learning_rate=0.0
        ),
        lambda: solution.train_som(
            np.ones((2, 1)),
            n_neurons=2,
            epochs=0,
            initial_learning_rate=0.1,
            final_learning_rate=0.2,
        ),
        lambda: solution.map_samples(np.ones((2, 2)), np.ones((2, 1))),
    ],
)
def test_invalid_som_inputs_are_rejected(call) -> None:
    with pytest.raises(ValueError):
        call()


def test_guided_demo_reports_training_improvement_and_mapping() -> None:
    result = subprocess.run(
        [sys.executable, str(TOPIC / "guided_demo.py")],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "X: (6, 1)" in result.stdout
    assert "prototypes: (4, 1)" in result.stdout
    assert "history: 41" in result.stdout
    assert "winners: (6,)" in result.stdout
