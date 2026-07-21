import importlib.util
from pathlib import Path
def test_reference():
    p=Path(__file__).resolve().parents[1]/"reference"/"solution.py"; s=importlib.util.spec_from_file_location("interval_ref",p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
    assert m.merge_intervals([(1,3),(2,6),(8,10),(10,12)])==[(1,6),(8,12)]
