import importlib.util
from pathlib import Path
def test_reference():
    p=Path(__file__).resolve().parents[1]/"reference"/"solution.py"; s=importlib.util.spec_from_file_location("prefix_ref",p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
    assert m.count_subarrays_with_sum([1,-1,0],0)==3
