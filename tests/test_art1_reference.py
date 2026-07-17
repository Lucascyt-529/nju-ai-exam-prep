import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "watermelon_book" / "05_neural_networks" / "08_art1"
SOLUTION = TOPIC / "reference" / "solution.py"


def load_solution_module():
    spec = importlib.util.spec_from_file_location("art1_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def sample_patterns() -> np.ndarray:
    return np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 0, 1, 1],
        ]
    )


def test_choice_scores_match_intersection_formula() -> None:
    pattern = np.array([1, 1, 0, 0])
    prototypes = np.array([[1, 1, 1, 0], [1, 0, 0, 0]])
    scores = solution.choice_scores(pattern, prototypes, alpha=1.0)
    np.testing.assert_allclose(scores, [2 / 4, 1 / 2])
    assert scores.shape == (2,)


def test_match_ratio_uses_pattern_size_denominator() -> None:
    pattern = np.array([1, 1, 0, 0])
    assert solution.match_ratio(pattern, np.array([1, 0, 1, 0])) == pytest.approx(0.5)


def test_selection_checks_candidates_in_score_order() -> None:
    pattern = np.array([1, 1, 1, 0])
    prototypes = np.array([[1, 0, 0, 0], [1, 1, 0, 1]])
    scores = solution.choice_scores(pattern, prototypes, alpha=0.1)
    assert scores[0] > scores[1]
    assert solution.select_resonant_category(
        pattern, prototypes, alpha=0.1, vigilance=0.6
    ) == 1


def test_equal_choice_scores_prefer_earlier_category() -> None:
    pattern = np.array([1, 1, 0])
    prototypes = np.array([[1, 0, 0], [0, 1, 0]])
    assert solution.select_resonant_category(
        pattern, prototypes, alpha=0.1, vigilance=0.5
    ) == 0


def test_low_vigilance_merges_first_two_patterns_by_intersection() -> None:
    prototypes, assignments, history = solution.train_art1(sample_patterns(), vigilance=0.4)
    assert prototypes.shape == (2, 4)
    np.testing.assert_array_equal(prototypes[0], [1, 0, 0, 0])
    np.testing.assert_array_equal(assignments, [0, 0, 1])
    assert history == [1, 1, 2]


def test_high_vigilance_creates_finer_categories() -> None:
    low = solution.train_art1(sample_patterns(), vigilance=0.4)[0]
    high = solution.train_art1(sample_patterns(), vigilance=0.75)[0]
    assert low.shape[0] == 2
    assert high.shape[0] == 3


def test_training_is_deterministic() -> None:
    X = sample_patterns()
    first = solution.train_art1(X, vigilance=0.4)
    second = solution.train_art1(X, vigilance=0.4)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert first[2] == second[2]


def test_training_does_not_modify_patterns() -> None:
    X = sample_patterns()
    original = X.copy()
    solution.train_art1(X, vigilance=0.4)
    np.testing.assert_array_equal(X, original)


def test_prediction_returns_minus_one_when_no_category_resonates() -> None:
    prototypes = np.array([[1, 1, 0, 0]])
    X = np.array([[0, 0, 1, 1]])
    np.testing.assert_array_equal(
        solution.predict_art1(X, prototypes, vigilance=0.5), [-1]
    )


def test_category_limit_failure_is_explicit() -> None:
    with pytest.raises(RuntimeError):
        solution.train_art1(sample_patterns(), vigilance=0.75, max_categories=2)


def test_input_order_can_change_online_prototypes() -> None:
    X = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
    forward = solution.train_art1(X, vigilance=0.4)[0]
    reverse = solution.train_art1(X[::-1], vigilance=0.4)[0]
    assert not np.array_equal(forward, reverse)


@pytest.mark.parametrize(
    "call",
    [
        lambda: solution.train_art1(np.array([1, 0, 1])),
        lambda: solution.train_art1(np.array([[0, 0, 0]])),
        lambda: solution.train_art1(np.array([[1, 2, 0]])),
        lambda: solution.train_art1(sample_patterns(), alpha=0.0),
        lambda: solution.train_art1(sample_patterns(), vigilance=-0.1),
        lambda: solution.train_art1(sample_patterns(), vigilance=1.1),
        lambda: solution.train_art1(sample_patterns(), max_categories=0),
        lambda: solution.match_ratio(np.array([0, 0]), np.array([1, 0])),
        lambda: solution.predict_art1(np.ones((1, 2)), np.ones((1, 3))),
    ],
)
def test_invalid_art1_inputs_are_rejected(call) -> None:
    with pytest.raises(ValueError):
        call()


def test_guided_demo_shows_vigilance_granularity() -> None:
    result = subprocess.run(
        [sys.executable, str(TOPIC / "guided_demo.py")],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "X: (3, 4)" in result.stdout
    assert "low vigilance: (2, 4) [0 0 1] [1, 1, 2]" in result.stdout
    assert "high vigilance: (3, 4) [0 1 2] [1, 2, 3]" in result.stdout
