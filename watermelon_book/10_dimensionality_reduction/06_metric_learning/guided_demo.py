"""引导演示：学习矩阵如何改变两个特征方向。"""
import importlib.util
from pathlib import Path
import numpy as np
SOLUTION=Path(__file__).resolve().parent/"reference"/"solution.py"
def main():
    spec=importlib.util.spec_from_file_location("metric_demo",SOLUTION)
    if spec is None or spec.loader is None: raise RuntimeError("无法加载参考实现")
    ml=importlib.util.module_from_spec(spec); spec.loader.exec_module(ml)
    X=np.array([[0.,0.],[0.,2.],[3.,0.],[3.,2.]]); y=np.array([0,0,1,1])
    model=ml.fit_pair_metric(X,y,margin=20,regularization=.1,learning_rate=.05,max_iterations=100)
    print("metric shape:",model["metric"].shape)
    print("metric:\n",np.round(model["metric"],4))
    print("minimum eigenvalue:",round(float(np.min(np.linalg.eigvalsh(model["metric"]))),8))
    print("objective first/last:",round(model["objective_history"][0],6),round(model["objective_history"][-1],6))
    print("x weight > y weight:",bool(model["metric"][0,0]>model["metric"][1,1]))
if __name__=="__main__": main()
