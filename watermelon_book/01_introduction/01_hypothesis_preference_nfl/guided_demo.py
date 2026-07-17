"""引导演示：同一训练集对应多个假设以及NFL平均。"""
import importlib.util
from pathlib import Path
import numpy as np
SOLUTION=Path(__file__).resolve().parent/"reference"/"solution.py"
def main():
    spec=importlib.util.spec_from_file_location("intro_demo",SOLUTION)
    if spec is None or spec.loader is None: raise RuntimeError("无法加载参考实现")
    intro=importlib.util.module_from_spec(spec); spec.loader.exec_module(intro)
    H=intro.enumerate_binary_hypotheses(4); V=intro.version_space(H,np.array([0,3]),np.array([0,1])); selected=intro.select_smooth_preference(V); nfl=intro.nfl_unseen_error_experiment(4,np.array([0,3]))
    print("hypothesis space shape:",H.shape)
    print("version space size:",len(V))
    print("preferred hypothesis:",selected.tolist())
    print("preferred transitions:",intro.transition_count(selected))
    print("NFL unseen errors:",nfl)
if __name__=="__main__": main()
