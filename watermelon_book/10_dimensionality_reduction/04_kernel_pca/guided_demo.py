"""引导演示：RBF核中心化、谱坐标和训练样本再投影。"""

import importlib.util
from pathlib import Path
import numpy as np

SOLUTION=Path(__file__).resolve().parent/"reference"/"solution.py"

def main():
    spec=importlib.util.spec_from_file_location("kpca_demo",SOLUTION)
    if spec is None or spec.loader is None: raise RuntimeError("无法加载参考实现")
    kpca=importlib.util.module_from_spec(spec); spec.loader.exec_module(kpca)
    X=np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.],[2.,0.],[0.,2.],[-2.,0.],[0.,-2.]])
    model=kpca.fit_kernel_pca(X,2,kernel="rbf",gamma=0.5)
    projected=kpca.transform_kernel_pca(model,X)
    print("kernel shape:",model["centered_kernel"].shape)
    print("kernel row means:",np.round(model["centered_kernel"].mean(axis=1),12).tolist())
    print("eigenvalues:",np.round(model["eigenvalues"],6).tolist())
    print("coordinates shape:",model["coordinates"].shape)
    print("training reprojection matches:",bool(np.allclose(projected,model["coordinates"])))

if __name__=="__main__": main()
