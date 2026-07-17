"""引导演示：精确状态分布和可复现Gibbs轨迹。"""
import importlib.util
from pathlib import Path
import numpy as np
SOLUTION=Path(__file__).resolve().parent/"reference"/"solution.py"
def main():
    spec=importlib.util.spec_from_file_location("bm_demo",SOLUTION)
    if spec is None or spec.loader is None: raise RuntimeError("无法加载参考实现")
    bm=importlib.util.module_from_spec(spec); spec.loader.exec_module(bm)
    W=np.array([[0.,1.,-.5],[1.,0.,.8],[-.5,.8,0.]]); b=np.array([.2,-.1,.3])
    model=bm.exact_distribution(W,b); trajectory=bm.gibbs_sample(np.array([0,0,0]),W,b,n_sweeps=5,random_state=4)
    print("states shape:",model["states"].shape)
    print("probability sum:",round(float(model["probabilities"].sum()),12))
    print("lowest-energy state:",model["states"][np.argmin(model["energies"])].tolist())
    print("most-probable state:",model["states"][np.argmax(model["probabilities"])].tolist())
    print("mean activations:",np.round(model["mean"],6).tolist())
    print("trajectory shape:",trajectory.shape)
if __name__=="__main__": main()
