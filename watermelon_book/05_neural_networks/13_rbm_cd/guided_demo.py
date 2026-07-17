"""引导演示：RBM一次正相、负相和CD-1更新。"""
import importlib.util
from pathlib import Path
import numpy as np
SOLUTION=Path(__file__).resolve().parent/"reference"/"solution.py"
def main():
    spec=importlib.util.spec_from_file_location("rbm_demo",SOLUTION)
    if spec is None or spec.loader is None: raise RuntimeError("无法加载参考实现")
    rbm=importlib.util.module_from_spec(spec); spec.loader.exec_module(rbm)
    V=np.array([[1,1,0,0],[1,0,0,0],[0,0,1,1],[0,0,0,1]])
    model=rbm.fit_rbm_cd1(V,2,epochs=20,learning_rate=.1,random_state=3)
    print("weight shape:",model["weights"].shape)
    print("hidden probability shape:",rbm.hidden_probabilities(V,model["weights"],model["hidden_bias"]).shape)
    print("reconstruction history shape:",model["reconstruction_history"].shape)
    print("parameters finite:",bool(np.all(np.isfinite(model["weights"]))))
    print("first/last reconstruction:",round(model["reconstruction_history"][0],6),round(model["reconstruction_history"][-1],6))
if __name__=="__main__": main()
