import importlib.util
from pathlib import Path
def test_reference():
    p=Path(__file__).resolve().parents[1]/"reference"/"solution.py"; s=importlib.util.spec_from_file_location("tree_ref",p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
    assert m.level_order(m.build_tree("3 9 20 # # 15 7".split()))==[[3],[9,20],[15,7]]
