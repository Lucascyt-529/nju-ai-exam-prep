"""引导演示：NCA降低噪声特征的距离权重。"""
import importlib.util
from pathlib import Path
import numpy as np
SOLUTION=Path(__file__).resolve().parent/"reference"/"solution.py"
def main():
    spec=importlib.util.spec_from_file_location("nca_demo",SOLUTION)
    if spec is None or spec.loader is None: raise RuntimeError("无法加载参考实现")
    nca=importlib.util.module_from_spec(spec); spec.loader.exec_module(nca)
    X=np.array([[-1.,-2.],[-.8,2.],[-.6,0.],[.6,-2.],[.8,2.],[1.,0.]]) ; y=np.array([0,0,0,1,1,1])
    model=nca.fit_nca(X,y,learning_rate=.2,max_iterations=200)
    print("probability shape:",model["probabilities"].shape)
    print("row sums:",np.round(model["probabilities"].sum(axis=1),12).tolist())
    print("objective first/last:",round(model["objective_history"][0],6),round(model["objective_history"][-1],6))
    print("metric diagonal:",np.round(np.diag(model["metric"]),6).tolist())
    print("metric minimum eigenvalue:",round(float(np.min(np.linalg.eigvalsh(model["metric"]))),10))
if __name__=="__main__": main()
