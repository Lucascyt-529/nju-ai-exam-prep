import importlib.util
from pathlib import Path
def test_reference():
    p=Path(__file__).resolve().parents[1]/"reference"/"solution.py"; s=importlib.util.spec_from_file_location("heap_ref",p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
    assert m.top_k_frequent([1,1,2,2,3],2)==[1,2]
