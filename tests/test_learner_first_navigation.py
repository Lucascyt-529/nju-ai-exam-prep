import re
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]

LEARNER_ENTRIES = {
    "01_numpy/README.md": [
        "../02_numpy_basics/00_array_creation_dtypes/README.md",
        "../02_numpy_basics/01_arrays_shapes_axes/README.md",
        "../02_numpy_basics/01_arrays_shapes_axes/reshape_practice/README.md",
        "../02_numpy_basics/02_indexing_filtering/README.md",
        "../02_numpy_basics/03_broadcasting/README.md",
        "../02_numpy_basics/04_matrix_multiplication/README.md",
    ],
    "03_data_io_processing/README.md": [
        "../00_python_programming/01_standard_input/README.md",
        "../00_python_programming/02_command_line_program/README.md",
        "../01_file_io/01_text_table/README.md",
        "../01_file_io/02_clean_csv/README.md",
        "../01_file_io/03_missing_csv/README.md",
        "../01_file_io/04_mixed_csv/README.md",
        "../01_file_io/05_numpy_model_files/README.md",
        "../03_data_processing/01_imputation_standardization/README.md",
        "../03_data_processing/02_categorical_encoding/README.md",
        "../03_data_processing/03_minibatch_training/README.md",
        "../03_data_processing/04_outlier_handling/README.md",
        "../04_pandas_basics/01_series_dataframe/README.md",
        "../04_pandas_basics/02_csv_selection/README.md",
        "../04_pandas_basics/03_cleaning/README.md",
        "../04_pandas_basics/04_groupby_merge/README.md",
        "../04_pandas_basics/05_pandas_numpy_bridge/README.md",
        "../04_pandas_basics/06_sort_pivot_melt/README.md",
    ],
    "04_algorithms/README.md": [
        f"../algorithms/{index:02d}_{name}/README.md"
        for index, name in enumerate(
            [
                "sorting_records",
                "binary_search_boundaries",
                "hashing_sequences",
                "graph_traversal",
                "grid_shortest_path",
                "dijkstra_shortest_path",
                "min_coin_change",
                "longest_common_subsequence",
                "zero_one_knapsack",
            ],
            start=1,
        )
    ],
    "05_integrated_practice/README.md": [
        f"../integrated_tasks/{index:02d}_{name}/README.md"
        for index, name in enumerate(
            [
                "preprocessing_pipeline",
                "linear_regression_csv",
                "logistic_regression_csv",
                "kmeans_csv",
                "mixed_feature_classification",
            ],
            start=1,
        )
    ],
    "06_mock_exams/README.md": [
        "../mock_exams/01_foundation_diagnostic/START_HERE.md",
        "../mock_exams/01_foundation_diagnostic/PAPER.md",
        "../mock_exams/02_transfer_integration/START_HERE.md",
        "../mock_exams/02_transfer_integration/PAPER.md",
    ],
}


def test_root_readme_exposes_six_stable_modules() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    entries = [
        "01_numpy/README.md",
        "02_machine_learning/README.md",
        "03_data_io_processing/README.md",
        "04_algorithms/README.md",
        "05_integrated_practice/README.md",
        "06_mock_exams/README.md",
    ]
    for entry in entries:
        assert entry in readme
    assert readme.count("**当前学习：线性回归**") == 1


def test_learner_catalogs_link_every_verified_topic() -> None:
    forbidden_placeholders = ("本轮不", "后续统一整理", "现有内容暂留", "后续训练重点")
    for catalog, expected_links in LEARNER_ENTRIES.items():
        text = (ROOT / catalog).read_text(encoding="utf-8")
        for link in expected_links:
            assert f"]({link})" in text, f"{catalog} 缺少 {link}"
        for phrase in forbidden_placeholders:
            assert phrase not in text, f"{catalog} 仍含占位措辞: {phrase}"


def test_legacy_indexes_redirect_to_learner_catalogs() -> None:
    redirects = {
        "00_python_programming/README.md": "../03_data_io_processing/README.md",
        "01_file_io/README.md": "../03_data_io_processing/README.md",
        "02_numpy_basics/README.md": "../01_numpy/README.md",
        "03_data_processing/README.md": "../03_data_io_processing/README.md",
        "04_pandas_basics/README.md": "../03_data_io_processing/README.md",
        "algorithms/README.md": "../04_algorithms/README.md",
        "integrated_tasks/README.md": "../05_integrated_practice/README.md",
        "mock_exams/README.md": "../06_mock_exams/README.md",
    }
    for legacy, target in redirects.items():
        text = (ROOT / legacy).read_text(encoding="utf-8")
        assert "兼容入口" in text
        assert f"]({target})" in text


def test_all_relative_markdown_links_resolve() -> None:
    link_pattern = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
    broken = []
    for markdown in ROOT.glob("**/*.md"):
        if any(part in {".git", ".pytest_cache"} for part in markdown.parts):
            continue
        text = markdown.read_text(encoding="utf-8")
        for raw_target in link_pattern.findall(text):
            target = raw_target.strip()
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            relative = unquote(target.split("#", maxsplit=1)[0])
            if relative and not (markdown.parent / relative).exists():
                broken.append(f"{markdown.relative_to(ROOT)} -> {target}")
    assert not broken, "相对链接失效:\n" + "\n".join(broken)
