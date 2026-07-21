import importlib.util
from pathlib import Path
def test_reference():
    p=Path(__file__).resolve().parents[1]/"reference"/"solution.py"; s=importlib.util.spec_from_file_location("topo_ref",p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
    assert m.course_order(4,[(1,0),(2,0),(3,1),(3,2)])==[0,1,2,3]; assert m.course_order(2,[(1,0),(0,1)])==[]
