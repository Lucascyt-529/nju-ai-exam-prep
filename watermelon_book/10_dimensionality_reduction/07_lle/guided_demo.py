"""引导演示：局部权重如何在低维空间中保留。"""
import importlib.util
from pathlib import Path
import numpy as np
SOLUTION=Path(__file__).resolve().parent/"reference"/"solution.py"
def main():
    spec=importlib.util.spec_from_file_location("lle_demo",SOLUTION)
    if spec is None or spec.loader is None: raise RuntimeError("无法加载参考实现")
    lle=importlib.util.module_from_spec(spec); spec.loader.exec_module(lle)
    t=np.linspace(-1.2,1.2,9); X=np.column_stack((t,t*t))
    model=lle.fit_lle(X,2,1)
    print("neighbor shape:",model["neighbors"].shape)
    print("weight shape:",model["weights"].shape)
    print("weight row sums:",np.round(model["weights"].sum(axis=1),12).tolist())
    print("coordinates shape:",model["coordinates"].shape)
    print("coordinate mean:",np.round(model["coordinates"].mean(axis=0),12).tolist())
    print("low-dimensional reconstruction error:",round(model["low_dimensional_error"],8))
if __name__=="__main__": main()
