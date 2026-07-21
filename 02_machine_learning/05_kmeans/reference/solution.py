"""参考实现：确定性边界规则的K-means++与K-means。"""

import numpy as np


def _matrix(X: np.ndarray, name: str) -> None:
    if not isinstance(X,np.ndarray) or X.ndim!=2 or 0 in X.shape or not np.issubdtype(X.dtype,np.number) or not np.all(np.isfinite(X)):
        raise ValueError(f"{name}必须是非空有限数值二维数组")


def _positive_int(value: object,name: str)->int:
    if not isinstance(value,(int,np.integer)) or isinstance(value,(bool,np.bool_)) or value<=0: raise ValueError(f"{name}必须是正整数")
    return int(value)


def squared_distance_matrix(X: np.ndarray,centers: np.ndarray)->np.ndarray:
    _matrix(X,"X"); _matrix(centers,"centers")
    if X.shape[1]!=centers.shape[1]: raise ValueError("X和centers特征数必须一致")
    difference=X.astype(float)[:,None,:]-centers.astype(float)[None,:,:]
    return np.sum(difference*difference,axis=2)


def _rng_seed(random_state: object)->int:
    if not isinstance(random_state,(int,np.integer)) or isinstance(random_state,(bool,np.bool_)): raise ValueError("random_state必须是整数")
    return int(random_state)


def kmeans_plus_plus(X: np.ndarray,n_clusters: int,*,random_state: int=0)->np.ndarray:
    _matrix(X,"X"); k=_positive_int(n_clusters,"n_clusters")
    if k>np.unique(X,axis=0).shape[0]: raise ValueError("n_clusters不能超过不同坐标点数量")
    rng=np.random.default_rng(_rng_seed(random_state)); first=int(rng.integers(0,X.shape[0])); centers=[X[first].astype(float).copy()]
    while len(centers)<k:
        distances=squared_distance_matrix(X,np.vstack(centers)); closest=np.min(distances,axis=1); total=float(np.sum(closest))
        if total<=0: raise RuntimeError("剩余K-means++采样权重为0")
        index=int(rng.choice(X.shape[0],p=closest/total)); centers.append(X[index].astype(float).copy())
    return np.vstack(centers)


def assign_labels(X: np.ndarray,centers: np.ndarray)->tuple[np.ndarray,np.ndarray]:
    distances=squared_distance_matrix(X,centers); labels=np.argmin(distances,axis=1)
    return labels,distances


def update_centers(X: np.ndarray,labels: np.ndarray,old_centers: np.ndarray)->np.ndarray:
    _matrix(X,"X"); _matrix(old_centers,"old_centers")
    if not isinstance(labels,np.ndarray) or labels.shape!=(X.shape[0],) or not np.issubdtype(labels.dtype,np.integer) or np.any(labels<0) or np.any(labels>=old_centers.shape[0]):
        raise ValueError("labels必须是合法簇下标")
    centers=np.empty_like(old_centers,dtype=float); empty=[]
    for cluster in range(old_centers.shape[0]):
        members=X[labels==cluster]
        if members.size: centers[cluster]=np.mean(members,axis=0)
        else: empty.append(cluster)
    if empty:
        distances=squared_distance_matrix(X,old_centers); assigned=distances[np.arange(X.shape[0]),labels]
        order=np.lexsort((np.arange(X.shape[0]),-assigned)); used=set()
        for cluster in empty:
            index=next(int(i) for i in order if int(i) not in used); used.add(index); centers[cluster]=X[index]
    return centers


def inertia(X: np.ndarray,centers: np.ndarray,labels: np.ndarray)->float:
    distances=squared_distance_matrix(X,centers)
    if not isinstance(labels,np.ndarray) or labels.shape!=(X.shape[0],): raise ValueError("labels形状无效")
    return float(np.sum(distances[np.arange(X.shape[0]),labels]))


def fit_kmeans(X: np.ndarray,n_clusters: int,*,random_state: int=0,max_iterations: int=100,tolerance: float=1e-6)->dict[str,object]:
    _matrix(X,"X"); k=_positive_int(n_clusters,"n_clusters"); limit=_positive_int(max_iterations,"max_iterations")
    if not isinstance(tolerance,(int,float,np.integer,np.floating)) or isinstance(tolerance,(bool,np.bool_)) or not np.isfinite(tolerance) or tolerance<0: raise ValueError("tolerance必须是非负有限数值")
    centers=kmeans_plus_plus(X,k,random_state=random_state); labels,_=assign_labels(X,centers); history=[inertia(X,centers,labels)]
    for iteration in range(1,limit+1):
        new_centers=update_centers(X,labels,centers); new_labels,_=assign_labels(X,new_centers); history.append(inertia(X,new_centers,new_labels))
        movement=float(np.max(np.linalg.norm(new_centers-centers,axis=1))); unchanged=np.array_equal(new_labels,labels)
        centers,labels=new_centers,new_labels
        if unchanged or movement<=float(tolerance): break
    return {"centers":centers,"labels":labels,"inertia_history":np.array(history),"iterations":iteration,"n_features":X.shape[1]}


def predict(model: dict[str,object],X: np.ndarray)->np.ndarray:
    _matrix(X,"X")
    if not isinstance(model,dict) or set(model)!={"centers","labels","inertia_history","iterations","n_features"} or X.shape[1]!=model["n_features"]: raise ValueError("model无效或特征数不匹配")
    return assign_labels(X,model["centers"])[0]
