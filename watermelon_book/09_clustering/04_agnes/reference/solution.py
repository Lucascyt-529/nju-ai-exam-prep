"""参考实现：single/complete/average链接AGNES。"""

from itertools import combinations
import numpy as np


LINKAGES={"single","complete","average"}


def _matrix(X: np.ndarray)->None:
    if not isinstance(X,np.ndarray) or X.ndim!=2 or 0 in X.shape or not np.issubdtype(X.dtype,np.number) or not np.all(np.isfinite(X)): raise ValueError("X必须是非空有限数值二维数组")


def pairwise_euclidean(X: np.ndarray)->np.ndarray:
    _matrix(X); difference=X.astype(float)[:,None,:]-X.astype(float)[None,:,:]
    return np.sqrt(np.sum(difference*difference,axis=2))


def cluster_distance(cluster_a: tuple[int,...],cluster_b: tuple[int,...],sample_distances: np.ndarray,linkage: str)->float:
    if linkage not in LINKAGES: raise ValueError("linkage必须是single、complete或average")
    if not isinstance(sample_distances,np.ndarray) or sample_distances.ndim!=2 or sample_distances.shape[0]!=sample_distances.shape[1] or sample_distances.shape[0]==0 or not np.all(np.isfinite(sample_distances)): raise ValueError("sample_distances必须是有限方阵")
    n=sample_distances.shape[0]
    for cluster in (cluster_a,cluster_b):
        if not isinstance(cluster,tuple) or not cluster or len(set(cluster))!=len(cluster) or any(not isinstance(i,(int,np.integer)) or isinstance(i,(bool,np.bool_)) or i<0 or i>=n for i in cluster): raise ValueError("簇必须是合法无重复下标元组")
    if set(cluster_a)&set(cluster_b): raise ValueError("两个簇不能重叠")
    values=sample_distances[np.ix_(cluster_a,cluster_b)]
    if linkage=="single": return float(np.min(values))
    if linkage=="complete": return float(np.max(values))
    return float(np.mean(values))


def fit_agnes(X: np.ndarray,n_clusters: int,*,linkage: str="average")->dict[str,object]:
    _matrix(X)
    if not isinstance(n_clusters,(int,np.integer)) or isinstance(n_clusters,(bool,np.bool_)) or n_clusters<=0 or n_clusters>X.shape[0]: raise ValueError("n_clusters必须位于1到n之间")
    if linkage not in LINKAGES: raise ValueError("linkage必须是single、complete或average")
    distances=pairwise_euclidean(X); active={i:(i,) for i in range(X.shape[0])}; next_id=X.shape[0]; history=[]
    while len(active)>int(n_clusters):
        best=None
        for left,right in combinations(sorted(active),2):
            distance=cluster_distance(active[left],active[right],distances,linkage); key=(distance,left,right)
            if best is None or key<best[0]: best=(key,left,right)
        key,left,right=best; members=tuple(sorted(active[left]+active[right])); del active[left]; del active[right]; active[next_id]=members
        history.append({"left":left,"right":right,"new":next_id,"distance":key[0],"members":members}); next_id+=1
    ordered=sorted(active.items(),key=lambda item:(min(item[1]),item[0])); labels=np.empty(X.shape[0],dtype=int)
    for label,(_,members) in enumerate(ordered): labels[list(members)]=label
    return {"labels":labels,"clusters":tuple(members for _,members in ordered),"history":tuple(history),"linkage":linkage,"sample_distances":distances}
