"""引导演示：K-means++中心、SSE历史与簇分配。"""

import importlib.util
from pathlib import Path
import numpy as np

SOLUTION=Path(__file__).resolve().parent/"reference"/"solution.py"

def main():
    spec=importlib.util.spec_from_file_location("kmeans_demo",SOLUTION)
    if spec is None or spec.loader is None: raise RuntimeError("无法加载参考实现")
    km=importlib.util.module_from_spec(spec); spec.loader.exec_module(km)
    X=np.array([[0.,0.],[0.,1.],[1.,0.],[8.,8.],[8.,9.],[9.,8.]])
    initial=km.kmeans_plus_plus(X,2,random_state=4); model=km.fit_kmeans(X,2,random_state=4)
    print("initial centers:",initial.tolist()); print("distance shape:",km.squared_distance_matrix(X,initial).shape)
    print("final centers:",np.round(model["centers"],4)); print("labels:",model["labels"].tolist())
    print("inertia history:",np.round(model["inertia_history"],6)); print("nonincreasing:",bool(np.all(np.diff(model["inertia_history"])<=1e-12)))

if __name__=="__main__": main()
