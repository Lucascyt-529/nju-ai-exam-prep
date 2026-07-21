import importlib.util
from pathlib import Path
def test_reference():
    p=Path(__file__).resolve().parents[1]/"reference"/"solution.py"; s=importlib.util.spec_from_file_location("two_pointer_ref",p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
    assert m.longest_unique_window("abba")== (2,0,2); assert m.longest_unique_window("")== (0,0,0)
