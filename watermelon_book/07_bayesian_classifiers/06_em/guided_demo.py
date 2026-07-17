"""引导演示：责任度软分配和EM似然历史。"""
import importlib.util
from pathlib import Path
import numpy as np
SOLUTION=Path(__file__).resolve().parent/"reference"/"solution.py"
def main():
    spec=importlib.util.spec_from_file_location("em_demo",SOLUTION)
    if spec is None or spec.loader is None: raise RuntimeError("无法加载参考实现")
    em=importlib.util.module_from_spec(spec); spec.loader.exec_module(em)
    heads=np.array([1,2,1,8,9,8]); tosses=np.full(6,10)
    model=em.fit_coin_mixture_em(heads,tosses,initial_probabilities=np.array([.25,.75]))
    print("responsibility shape:",model["responsibilities"].shape)
    print("mixing:",np.round(model["mixing"],6).tolist())
    print("coin probabilities:",np.round(model["probabilities"],6).tolist())
    print("likelihood nondecreasing:",bool(np.all(np.diff(model["log_likelihood_history"])>=-1e-10)))
    step=em.em_step_diagnostics(heads,tosses,np.array([.5,.5]),np.array([.25,.75]))
    print("old likelihood equals tight bound:",bool(np.isclose(step["old_likelihood"],step["old_bound"])))
    print("M step raises fixed-responsibility bound:",bool(step["after_m_bound"]>=step["old_bound"]-1e-12))
    print("new likelihood equals new tight bound:",bool(np.isclose(step["new_likelihood"],step["tight_new_bound"])))
    print("iterations:",model["iterations"])
if __name__=="__main__": main()
