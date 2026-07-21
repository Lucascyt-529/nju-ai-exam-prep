import importlib.util
from pathlib import Path
def test_reference():
    p=Path(__file__).resolve().parents[1]/"reference"/"solution.py"; s=importlib.util.spec_from_file_location("list_ref",p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
    assert m.linked_list_values(m.reverse_linked_list(m.build_linked_list([1,2,3])))==[3,2,1]
