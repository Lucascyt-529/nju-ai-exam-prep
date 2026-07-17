"""引导演示：距离矩阵、轮廓值与外部指标。"""

import importlib.util
from pathlib import Path
import numpy as np

SOLUTION=Path(__file__).resolve().parent/"reference"/"solution.py"

def main():
    spec=importlib.util.spec_from_file_location("cluster_metrics_demo",SOLUTION)
    if spec is None or spec.loader is None: raise RuntimeError("无法加载参考实现")
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    X=np.array([[0.,0.],[0.,1.],[5.,5.],[5.,6.],[10.,10.]])
    labels=np.array([10,10,30,30,50]); alternative=np.array([1,1,2,2,2])
    print("distance shape:",m.pairwise_minkowski(X,X).shape)
    print("SSE:",m.within_cluster_sse(X,labels))
    print("silhouette:",np.round(m.silhouette_samples(X,labels),4))
    print("pair counts:",m.pair_confusion_counts(labels,alternative))
    print("Rand/ARI:",round(m.rand_index(labels,alternative),4),round(m.adjusted_rand_index(labels,alternative),4))

if __name__=="__main__": main()
