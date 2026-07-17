"""引导演示：曲线路径的直线距离、测地距离与Isomap。"""
import importlib.util
from pathlib import Path
import numpy as np
SOLUTION=Path(__file__).resolve().parent/"reference"/"solution.py"
def main():
    spec=importlib.util.spec_from_file_location("isomap_demo",SOLUTION)
    if spec is None or spec.loader is None: raise RuntimeError("无法加载参考实现")
    iso=importlib.util.module_from_spec(spec); spec.loader.exec_module(iso)
    angles=np.linspace(0,np.pi,7); X=np.column_stack((np.cos(angles),np.sin(angles)))
    model=iso.fit_isomap(X,2,1)
    print("graph shape:",model["graph"].shape)
    print("finite undirected edges:",int(np.count_nonzero(np.triu(np.isfinite(model["graph"]),1))))
    print("endpoint euclidean:",round(model["euclidean_distances"][0,-1],6))
    print("endpoint geodesic:",round(model["geodesic_distances"][0,-1],6))
    print("coordinates shape:",model["coordinates"].shape)
    print("coordinates centered:",bool(np.allclose(model["coordinates"].mean(axis=0),0)))
if __name__=="__main__": main()
