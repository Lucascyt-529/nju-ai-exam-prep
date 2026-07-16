import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
COIN = ROOT / "algorithms" / "07_min_coin_change"
LCS = ROOT / "algorithms" / "08_longest_common_subsequence"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


coin = load_module("coin_solution", COIN / "reference" / "solution.py")
lcs = load_module("lcs_solution", LCS / "reference" / "solution.py")


def test_coin_sample_dp_and_reconstruction() -> None:
    minimum, combination, dp = coin.minimum_coin_change([1, 5, 7], 11)
    assert minimum == 3 and combination == [1, 5, 5]
    assert len(dp) == 12 and dp[0] == 0 and dp[11] == 3


def test_coin_zero_unreachable_and_tie_rule() -> None:
    assert coin.minimum_coin_change([2, 5], 0)[:2] == (0, [])
    assert coin.minimum_coin_change([2, 4], 3)[:2] == (-1, [])
    assert coin.minimum_coin_change([1, 2, 3], 4)[:2] == (2, [1, 3])


def test_coin_input_is_not_modified() -> None:
    coins = [5, 1, 7]
    original = coins.copy()
    coin.minimum_coin_change(coins, 11)
    assert coins == original


def test_coin_sample_and_cli_modes(tmp_path: Path) -> None:
    source = (COIN / "data" / "sample_input.txt").read_text(encoding="utf-8")
    expected = (COIN / "expected" / "sample_output.txt").read_text(encoding="utf-8")
    assert coin.solve(source) == expected.rstrip("\n")
    stdin_run = subprocess.run(
        [sys.executable, str(COIN / "reference" / "solution.py")], input=source, text=True, capture_output=True, check=True
    )
    assert stdin_run.stdout == expected
    output = tmp_path / "coin" / "answer.txt"
    subprocess.run(
        [sys.executable, str(COIN / "reference" / "solution.py"), str(COIN / "data" / "sample_input.txt"), str(output)], check=True
    )
    assert output.read_bytes() == (COIN / "expected" / "sample_output.txt").read_bytes()


@pytest.mark.parametrize(
    "text",
    ["", "2 5\n1\n", "2 -1\n1 2\n", "2 5\n1 1\n", "2 5\n0 2\n", "x 5\n1 2\n"],
)
def test_invalid_coin_inputs_are_rejected(text: str) -> None:
    with pytest.raises(ValueError):
        coin.parse_problem(text)


def test_lcs_table_shape_length_and_deterministic_sequence() -> None:
    first, second = "ABCBDAB", "BDCABA"
    table = lcs.lcs_table(first, second)
    assert len(table) == len(first) + 1 and len(table[0]) == len(second) + 1
    assert table[-1][-1] == 4
    assert lcs.reconstruct_lcs(first, second, table) == "BCBA"


def test_lcs_empty_sequences_and_no_common_character() -> None:
    assert lcs.solve("\n\n") == "length: 0\nsequence: "
    assert lcs.solve("ABC\nXYZ\n") == "length: 0\nsequence: "


def test_lcs_tie_rule_is_deterministic() -> None:
    assert lcs.solve("ABC\nBAC\n") == "length: 2\nsequence: AC"


def test_lcs_sample_and_cli_modes(tmp_path: Path) -> None:
    source = (LCS / "data" / "sample_input.txt").read_text(encoding="utf-8")
    expected = (LCS / "expected" / "sample_output.txt").read_text(encoding="utf-8")
    assert lcs.solve(source) == expected.rstrip("\n")
    stdin_run = subprocess.run(
        [sys.executable, str(LCS / "reference" / "solution.py")], input=source, text=True, capture_output=True, check=True
    )
    assert stdin_run.stdout == expected
    output = tmp_path / "lcs" / "answer.txt"
    subprocess.run(
        [sys.executable, str(LCS / "reference" / "solution.py"), str(LCS / "data" / "sample_input.txt"), str(output)], check=True
    )
    assert output.read_bytes() == (LCS / "expected" / "sample_output.txt").read_bytes()


@pytest.mark.parametrize(
    "text",
    ["only-one-line", "A\nB\nC\n", "A B\nAB\n"],
)
def test_invalid_lcs_inputs_are_rejected(text: str) -> None:
    with pytest.raises(ValueError):
        lcs.parse_sequences(text)


def test_lcs_rejects_wrong_table_shape() -> None:
    with pytest.raises(ValueError):
        lcs.reconstruct_lcs("A", "A", [[0]])
