import importlib.util
from pathlib import Path
def test_reference():
    p=Path(__file__).resolve().parents[1]/"reference"/"solution.py"; s=importlib.util.spec_from_file_location("stack_ref",p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
    assert m.days_until_warmer([30,40,35,50])==[1,2,1,0]
