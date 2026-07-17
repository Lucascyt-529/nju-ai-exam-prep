"""引导演示：二维椭圆簇的软责任度与EM。"""
import importlib.util
from pathlib import Path
import numpy as np
SOLUTION=Path(__file__).resolve().parent/"reference"/"solution.py"
def main():
    spec=importlib.util.spec_from_file_location("gmm_demo",SOLUTION)
    if spec is None or spec.loader is None: raise RuntimeError("无法加载参考实现")
    gmm=importlib.util.module_from_spec(spec); spec.loader.exec_module(gmm)
    X=np.array([[-3.,0.],[-2.8,.2],[-3.2,-.1],[3.,0.],[2.8,-.2],[3.2,.1]])
    model=gmm.fit_gaussian_mixture(X,2,random_state=4)
    print("responsibility shape:",model["responsibilities"].shape)
    print("weights:",np.round(model["weights"],6).tolist())
    print("means:\n",np.round(model["means"],4))
    print("labels:",model["labels"].tolist())
    print("likelihood nondecreasing:",bool(np.all(np.diff(model["log_likelihood_history"])>=-1e-8)))
if __name__=="__main__": main()
