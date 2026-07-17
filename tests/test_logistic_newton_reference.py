import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "watermelon_book" / "03_linear_models" / "02_logistic_regression"
SOLUTION = TOPIC / "reference" / "solution.py"
STARTER = TOPIC / "starter.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_module(SOLUTION, "logistic_newton_solution")
starter = load_module(STARTER, "logistic_newton_starter")


def training_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[-2.0], [-1.0], [-0.5], [0.5], [1.0], [2.0]])
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    return X, y


def gradient_vector(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
    l2: float,
) -> np.ndarray:
    gradient_weights, gradient_bias = solution.logistic_gradients(
        X, y, weights, bias, l2=l2
    )
    return np.concatenate((gradient_weights, [gradient_bias]))


def test_hessian_shape_symmetry_and_positive_semidefinite_property() -> None:
    X = np.array([[1.0, 2.0], [2.0, -1.0], [0.0, 1.0], [-1.0, -2.0]])
    y = np.array([1.0, 1.0, 0.0, 0.0])
    hessian = solution.logistic_hessian(
        X, y, np.array([0.2, -0.3]), 0.1, l2=0.4
    )
    assert hessian.shape == (3, 3)
    np.testing.assert_allclose(hessian, hessian.T, atol=1e-15)
    assert np.min(np.linalg.eigvalsh(hessian)) >= -1e-14


def test_hessian_matches_finite_differences_of_full_gradient() -> None:
    X = np.array([[1.0, 2.0], [2.0, -1.0], [0.0, 1.0], [-1.0, -2.0]])
    y = np.array([1.0, 1.0, 0.0, 0.0])
    weights = np.array([0.2, -0.3])
    bias = 0.1
    l2 = 0.4
    hessian = solution.logistic_hessian(X, y, weights, bias, l2=l2)
    beta = np.concatenate((weights, [bias]))
    epsilon = 1e-5
    numerical = np.empty_like(hessian)
    for column in range(beta.size):
        offset = np.zeros_like(beta)
        offset[column] = epsilon
        plus = gradient_vector(X, y, beta[:-1] + offset[:-1], beta[-1] + offset[-1], l2)
        minus = gradient_vector(X, y, beta[:-1] - offset[:-1], beta[-1] - offset[-1], l2)
        numerical[:, column] = (plus - minus) / (2.0 * epsilon)
    np.testing.assert_allclose(hessian, numerical, rtol=1e-6, atol=1e-8)


def test_l2_is_added_to_weight_diagonal_but_not_bias_diagonal() -> None:
    X, y = training_data()
    weights = np.array([0.3])
    without_l2 = solution.logistic_hessian(X, y, weights, 0.2, l2=0.0)
    with_l2 = solution.logistic_hessian(X, y, weights, 0.2, l2=0.7)
    difference = with_l2 - without_l2
    np.testing.assert_allclose(difference, [[0.7, 0.0], [0.0, 0.0]], atol=1e-15)


def test_newton_direction_solves_damped_linear_system() -> None:
    X, y = training_data()
    weights = np.array([0.2])
    bias = -0.1
    l2 = 0.3
    damping = 0.05
    direction = solution.newton_direction(
        X, y, weights, bias, l2=l2, damping=damping
    )
    hessian = solution.logistic_hessian(X, y, weights, bias, l2=l2)
    gradient = gradient_vector(X, y, weights, bias, l2)
    np.testing.assert_allclose(
        (hessian + damping * np.eye(2)) @ direction,
        gradient,
        rtol=1e-12,
        atol=1e-12,
    )


def test_one_newton_step_matches_manual_full_step_when_accepted() -> None:
    X, y = training_data()
    initial_weights = np.array([0.0])
    initial_bias = 0.0
    direction = solution.newton_direction(
        X, y, initial_weights, initial_bias, l2=0.1, damping=1e-6
    )
    weights, bias, losses = solution.fit_newton(
        X,
        y,
        n_steps=1,
        l2=0.1,
        damping=1e-6,
        initial_weights=initial_weights,
        initial_bias=initial_bias,
    )
    np.testing.assert_allclose(weights, initial_weights - direction[:-1])
    assert bias == pytest.approx(initial_bias - direction[-1])
    assert losses[1] < losses[0]


def test_newton_training_learns_data_with_nonincreasing_loss() -> None:
    X, y = training_data()
    weights, bias, losses = solution.fit_newton(
        X, y, n_steps=6, l2=0.01, damping=1e-8
    )
    assert losses.shape == (7,)
    assert np.all(np.diff(losses) <= 1e-15)
    probabilities = solution.predict_proba(X, weights, bias)
    np.testing.assert_array_equal(solution.predict_labels(probabilities), y.astype(int))


def test_few_newton_steps_reach_lower_loss_than_more_gradient_steps_on_demo() -> None:
    X, y = training_data()
    _, _, newton_losses = solution.fit_newton(
        X, y, n_steps=6, l2=0.01, damping=1e-8
    )
    _, _, gradient_losses = solution.fit_gradient_descent(
        X, y, learning_rate=0.2, n_steps=100, l2=0.01
    )
    assert newton_losses[-1] < gradient_losses[-1]
    assert len(newton_losses) == 7 and len(gradient_losses) == 101


def test_rank_deficient_design_requires_damping_when_unregularized() -> None:
    first = np.array([-2.0, -1.0, 1.0, 2.0])
    X = np.column_stack((first, first))
    y = np.array([0.0, 0.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="Hessian奇异"):
        solution.newton_direction(
            X, y, np.zeros(2), 0.0, l2=0.0, damping=0.0
        )
    direction = solution.newton_direction(
        X, y, np.zeros(2), 0.0, l2=0.0, damping=1e-4
    )
    assert direction.shape == (3,) and np.all(np.isfinite(direction))


def test_damped_training_handles_rank_deficient_duplicate_features() -> None:
    first = np.array([-2.0, -1.0, 1.0, 2.0])
    X = np.column_stack((first, first))
    y = np.array([0.0, 0.0, 1.0, 1.0])
    weights, bias, losses = solution.fit_newton(
        X, y, n_steps=5, damping=1e-4, l2=0.0
    )
    assert np.all(np.isfinite(weights)) and np.isfinite(bias)
    assert losses[-1] < losses[0]


def test_extreme_saturated_probabilities_need_damping_for_direction() -> None:
    X, y = training_data()
    weights = np.array([1000.0])
    hessian = solution.logistic_hessian(X, y, weights, 0.0, l2=0.0)
    assert np.all(np.isfinite(hessian))
    direction = solution.newton_direction(
        X, y, weights, 0.0, l2=0.0, damping=1e-3
    )
    assert np.all(np.isfinite(direction))


def test_training_is_reproducible_and_does_not_modify_initial_weights() -> None:
    X, y = training_data()
    initial = np.array([0.3])
    original = initial.copy()
    first = solution.fit_newton(
        X, y, n_steps=4, l2=0.1, damping=1e-6, initial_weights=initial
    )
    second = solution.fit_newton(
        X, y, n_steps=4, l2=0.1, damping=1e-6, initial_weights=initial
    )
    np.testing.assert_array_equal(initial, original)
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1] == second[1]
    np.testing.assert_array_equal(first[2], second[2])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_steps": 0},
        {"n_steps": True},
        {"n_steps": 1, "damping": -1.0},
        {"n_steps": 1, "damping": True},
        {"n_steps": 1, "step_size": 0.0},
        {"n_steps": 1, "step_size": 1.1},
        {"n_steps": 1, "max_backtracking": -1},
        {"n_steps": 1, "max_backtracking": True},
    ],
)
def test_invalid_newton_hyperparameters_are_rejected(kwargs) -> None:
    X, y = training_data()
    with pytest.raises(ValueError):
        solution.fit_newton(X, y, **kwargs)


@pytest.mark.parametrize(
    "call",
    [
        lambda: solution.logistic_hessian(
            np.ones((2, 1)), np.ones((2, 1)), np.ones(1), 0.0
        ),
        lambda: solution.logistic_hessian(
            np.ones((2, 1)), np.array([0.0, 1.0]), np.ones(2), 0.0
        ),
        lambda: solution.newton_direction(
            np.ones((2, 1)), np.array([0.0, 1.0]), np.ones(1), np.nan
        ),
    ],
)
def test_invalid_newton_data_and_parameter_shapes_are_rejected(call) -> None:
    with pytest.raises((ValueError, TypeError)):
        call()


@pytest.mark.parametrize(
    "function_name",
    ["logistic_hessian", "newton_direction", "fit_newton"],
)
def test_student_newton_entry_points_remain_unimplemented(function_name: str) -> None:
    X, y = training_data()
    function = getattr(starter, function_name)
    with pytest.raises(NotImplementedError):
        if function_name == "logistic_hessian":
            function(X, y, np.zeros(1), 0.0)
        elif function_name == "newton_direction":
            function(X, y, np.zeros(1), 0.0)
        else:
            function(X, y, n_steps=1)


def test_guided_demo_reports_hessian_and_optimizer_comparison() -> None:
    result = subprocess.run(
        [sys.executable, str(TOPIC / "guided_demo.py")],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "Hessian shape: (3, 3)" in result.stdout
    assert "Hessian symmetric: True" in result.stdout
    assert "gradient descent: steps/final loss: 100" in result.stdout
    assert "Newton: steps/final loss: 6" in result.stdout
    assert "Newton non-increasing: True" in result.stdout
