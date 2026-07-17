"""引导演示：三种链接的合并历史。"""

import importlib.util
from pathlib import Path
import numpy as np

SOLUTION=Path(__file__).resolve().parent/"reference"/"solution.py"

def main():
    spec=importlib.util.spec_from_file_location("agnes_demo",SOLUTION)
    if spec is None or spec.loader is None: raise RuntimeError("无法加载参考实现")
    ag=importlib.util.module_from_spec(spec); spec.loader.exec_module(ag)
    X=np.array([[0.],[1.],[4.],[5.],[10.]])
    for linkage in ("single","complete","average"):
        model=ag.fit_agnes(X,2,linkage=linkage)
        print("linkage:",linkage,"clusters:",model["clusters"],"labels:",model["labels"].tolist())
        print("merge distances:",[round(step["distance"],4) for step in model["history"]])

if __name__=="__main__": main()
