from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_control_files_exist() -> None:
    required_files = {
        Path("README.md"),
        Path("START_HERE.md"),
        Path("AGENTS.md"),
        Path("CONTRIBUTING.md"),
        Path("LEARNING_STATUS.md"),
        Path("docs/project_management/BUILD_STATUS.md"),
        Path("docs/project_management/historical_baseline.md"),
        Path("docs/project_management/legacy_progress.md"),
        Path("docs/curriculum/learning_method.md"),
        Path("docs/curriculum/syllabus.md"),
        Path("docs/curriculum/full_coverage_matrix.md"),
        Path("requirements.txt"),
        Path("pytest.ini"),
    }

    missing = sorted(str(path) for path in required_files if not (ROOT / path).is_file())

    assert not missing, f"缺少仓库基础文件: {missing}"


def test_moved_management_files_are_not_duplicated_at_root() -> None:
    old_root_files = {
        "PROJECT_STATE.md",
        "progress.md",
        "coverage_matrix.md",
        "foundation_coverage.md",
        "historical_baseline.md",
        "learning_method.md",
        "syllabus.md",
        "exam_evidence.md",
    }
    duplicates = sorted(name for name in old_root_files if (ROOT / name).exists())
    assert not duplicates, f"管理文档不应在根目录保留重复副本: {duplicates}"


def test_markdown_files_are_valid_utf8() -> None:
    markdown_files = ROOT.glob("**/*.md")

    for path in markdown_files:
        text = path.read_text(encoding="utf-8")
        assert text.strip(), f"{path.name} 不能为空"


def test_coverage_matrix_contains_all_sixteen_chapters() -> None:
    matrix = (ROOT / "docs/curriculum/full_coverage_matrix.md").read_text(
        encoding="utf-8"
    )

    missing = [chapter for chapter in range(1, 17) if f"第{chapter}章" not in matrix]

    assert not missing, f"覆盖矩阵缺少章节: {missing}"


def test_root_readme_has_one_current_learning_entry() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    expected = "02_machine_learning/01_linear_regression/README.md"
    assert expected in readme
    assert readme.count("**当前学习：线性回归**") == 1

    legacy_entry = (ROOT / "START_HERE.md").read_text(encoding="utf-8")
    assert "本文件只为旧链接保留" in legacy_entry


def test_reference_solutions_have_no_placeholders() -> None:
    reference_files = sorted(ROOT.glob("**/reference/solution.py"))

    assert reference_files, "至少需要一个经过测试的参考实现"
    for path in reference_files:
        source = path.read_text(encoding="utf-8")
        assert "NotImplementedError" not in source, f"{path} 仍含未实现占位"
        assert "TODO" not in source, f"{path} 仍含 TODO"


def test_core_reference_solutions_do_not_use_forbidden_libraries() -> None:
    forbidden_imports = ("sklearn", "pandas", "scipy", "torch", "tensorflow", "jax")
    core_roots = [
        ROOT / "00_python_programming",
        ROOT / "01_file_io",
        ROOT / "02_numpy_basics",
        ROOT / "03_data_processing",
    ]

    violations = []
    for core_root in core_roots:
        for path in core_root.glob("**/reference/solution.py"):
            source = path.read_text(encoding="utf-8")
            for library in forbidden_imports:
                if f"import {library}" in source or f"from {library}" in source:
                    violations.append(f"{path}: {library}")

    assert not violations, f"核心参考实现使用了禁用库: {violations}"
