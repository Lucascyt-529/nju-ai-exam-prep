import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "watermelon_book" / "05_neural_networks" / "04_bp_training"
SOLUTION = TOPIC / "reference" / "solution.py"
STARTER = TOPIC / "starter.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_module(SOLUTION, "bp_mode_solution")
starter = load_module(STARTER, "bp_mode_starter")


def linear_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[-2.0, -1.0], [-1.0, -2.0], [1.0, 1.0], [2.0, 1.0]])
    y = np.array([[0.0], [0.0], [1.0], [1.0]])
    return X, y


def assert_same_parameters(
    first: dict[str, np.ndarray], second: dict[str, np.ndarray]
) -> None:
    assert set(first) == set(second)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_unshuffled_orders_visit_every_sample_in_input_order() -> None:
    orders = solution.make_epoch_sample_orders(4, 3)
    assert len(orders) == 3
    for order in orders:
        np.testing.assert_array_equal(order, np.arange(4))


def test_shuffled_orders_use_one_reproducible_generator_across_epochs() -> None:
    first = solution.make_epoch_sample_orders(6, 4, shuffle=True, random_state=8)
    second = solution.make_epoch_sample_orders(6, 4, shuffle=True, random_state=8)
    assert len(first) == 4
    for first_order, second_order in zip(first, second):
        np.testing.assert_array_equal(np.sort(first_order), np.arange(6))
        np.testing.assert_array_equal(first_order, second_order)
    assert any(not np.array_equal(first[0], order) for order in first[1:])


def test_zero_epochs_produces_no_orders() -> None:
    assert solution.make_epoch_sample_orders(3, 0, shuffle=True) == ()


@pytest.mark.parametrize(
    "args, kwargs",
    [
        ((0, 1), {}),
        ((True, 1), {}),
        ((3, -1), {}),
        ((3, 1.5), {}),
        ((3, 1), {"shuffle": 1}),
        ((3, 1), {"random_state": True}),
    ],
)
def test_invalid_order_arguments_are_rejected(args, kwargs) -> None:
    with pytest.raises(ValueError):
        solution.make_epoch_sample_orders(*args, **kwargs)


def test_accumulated_bp_named_entry_matches_legacy_train_network() -> None:
    X, y = linear_data()
    expected_parameters, expected_history = solution.train_network(
        X, y, n_hidden=3, learning_rate=0.2, epochs=5, seed=4
    )
    actual_parameters, actual_history = solution.train_network_accumulated_bp(
        X, y, n_hidden=3, learning_rate=0.2, epochs=5, seed=4
    )
    assert actual_history == expected_history
    assert_same_parameters(actual_parameters, expected_parameters)


def test_one_sample_standard_and_accumulated_bp_are_identical() -> None:
    X = np.array([[1.5, -0.5]])
    y = np.array([[1.0]])
    standard_parameters, standard_history = solution.train_network_standard_bp(
        X, y, n_hidden=2, learning_rate=0.1, epochs=4, seed=3, shuffle=True
    )
    accumulated_parameters, accumulated_history = solution.train_network_accumulated_bp(
        X, y, n_hidden=2, learning_rate=0.1, epochs=4, seed=3
    )
    assert standard_history == accumulated_history
    assert_same_parameters(standard_parameters, accumulated_parameters)


def test_standard_bp_history_is_epoch_aligned_and_learns_linear_data() -> None:
    X, y = linear_data()
    parameters, history = solution.train_network_standard_bp(
        X,
        y,
        n_hidden=3,
        learning_rate=0.1,
        epochs=200,
        seed=2,
        shuffle=True,
        random_state=7,
    )
    assert len(history) == 201
    assert history[-1] < history[0] * 0.05
    np.testing.assert_array_equal(solution.predict_labels(X, parameters), y.astype(int))


def test_standard_and_accumulated_bp_follow_different_parameter_paths() -> None:
    X, y = linear_data()
    standard_parameters, standard_history = solution.train_network_standard_bp(
        X, y, n_hidden=3, learning_rate=0.2, epochs=2, seed=2
    )
    accumulated_parameters, accumulated_history = solution.train_network_accumulated_bp(
        X, y, n_hidden=3, learning_rate=0.2, epochs=2, seed=2
    )
    assert standard_history[0] == accumulated_history[0]
    assert not np.allclose(standard_parameters["W1"], accumulated_parameters["W1"])
    assert standard_history != accumulated_history


def test_standard_bp_is_order_sensitive_but_accumulated_bp_is_row_invariant() -> None:
    X, y = linear_data()
    standard_forward, _ = solution.train_network_standard_bp(
        X, y, n_hidden=3, learning_rate=0.3, epochs=1, seed=2
    )
    standard_reverse, _ = solution.train_network_standard_bp(
        X[::-1].copy(), y[::-1].copy(), n_hidden=3, learning_rate=0.3, epochs=1, seed=2
    )
    assert not np.allclose(standard_forward["W1"], standard_reverse["W1"])

    accumulated_forward, _ = solution.train_network_accumulated_bp(
        X, y, n_hidden=3, learning_rate=0.3, epochs=1, seed=2
    )
    accumulated_reverse, _ = solution.train_network_accumulated_bp(
        X[::-1].copy(), y[::-1].copy(), n_hidden=3, learning_rate=0.3, epochs=1, seed=2
    )
    for key in accumulated_forward:
        np.testing.assert_allclose(
            accumulated_forward[key], accumulated_reverse[key], rtol=0.0, atol=1e-15
        )


def test_shuffled_standard_bp_is_reproducible_from_both_seeds() -> None:
    X, y = linear_data()
    first_parameters, first_history = solution.train_network_standard_bp(
        X,
        y,
        n_hidden=3,
        learning_rate=0.1,
        epochs=10,
        seed=5,
        shuffle=True,
        random_state=11,
    )
    second_parameters, second_history = solution.train_network_standard_bp(
        X,
        y,
        n_hidden=3,
        learning_rate=0.1,
        epochs=10,
        seed=5,
        shuffle=True,
        random_state=11,
    )
    assert first_history == second_history
    assert_same_parameters(first_parameters, second_parameters)


def test_standard_bp_does_not_modify_training_arrays() -> None:
    X, y = linear_data()
    original_X, original_y = X.copy(), y.copy()
    solution.train_network_standard_bp(
        X, y, n_hidden=3, learning_rate=0.1, epochs=3, seed=0, shuffle=True
    )
    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(y, original_y)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_hidden": 0, "learning_rate": 0.1, "epochs": 1},
        {"n_hidden": 2, "learning_rate": 0.0, "epochs": 1},
        {"n_hidden": 2, "learning_rate": 0.1, "epochs": -1},
        {"n_hidden": 2, "learning_rate": 0.1, "epochs": 1, "shuffle": 1},
        {"n_hidden": 2, "learning_rate": 0.1, "epochs": 1, "random_state": True},
    ],
)
def test_invalid_standard_bp_hyperparameters_are_rejected(kwargs) -> None:
    X, y = linear_data()
    with pytest.raises(ValueError):
        solution.train_network_standard_bp(X, y, **kwargs)


@pytest.mark.parametrize(
    "function_name",
    [
        "make_epoch_sample_orders",
        "train_network_accumulated_bp",
        "train_network_standard_bp",
    ],
)
def test_student_entry_points_remain_unimplemented(function_name: str) -> None:
    function = getattr(starter, function_name)
    if function_name == "make_epoch_sample_orders":
        arguments = (4, 1)
        kwargs = {}
    else:
        arguments = linear_data()
        kwargs = {"n_hidden": 2, "learning_rate": 0.1, "epochs": 1}
    with pytest.raises(NotImplementedError):
        function(*arguments, **kwargs)
