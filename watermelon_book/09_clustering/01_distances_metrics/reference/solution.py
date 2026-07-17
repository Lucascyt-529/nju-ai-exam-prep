"""参考实现：距离、SSE、轮廓系数、Rand与ARI。"""

import numpy as np


def _numeric_matrix(X: np.ndarray, name: str) -> None:
    if not isinstance(X, np.ndarray) or X.ndim != 2 or 0 in X.shape or not np.issubdtype(X.dtype, np.number) or not np.all(np.isfinite(X)):
        raise ValueError(f"{name}必须是非空有限数值二维数组")


def _labels(labels: np.ndarray, n: int, name: str) -> None:
    if not isinstance(labels, np.ndarray) or labels.shape != (n,) or labels.size == 0:
        raise ValueError(f"{name}必须是形状(n,)的一维数组")
    if labels.dtype.kind in "fc" and not np.all(np.isfinite(labels)):
        raise ValueError(f"{name}不能包含非有限值")


def pairwise_minkowski(X: np.ndarray, Z: np.ndarray, *, p: float = 2.0) -> np.ndarray:
    _numeric_matrix(X, "X"); _numeric_matrix(Z, "Z")
    if X.shape[1] != Z.shape[1]: raise ValueError("X和Z特征数必须一致")
    if not isinstance(p, (int,float,np.integer,np.floating)) or isinstance(p,(bool,np.bool_)) or not np.isfinite(p) or p < 1:
        raise ValueError("p必须是不小于1的有限数值")
    return np.sum(np.abs(X.astype(float)[:,None,:]-Z.astype(float)[None,:,:])**float(p),axis=2)**(1.0/float(p))


def pairwise_hamming(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    if not isinstance(X,np.ndarray) or not isinstance(Z,np.ndarray) or X.ndim!=2 or Z.ndim!=2 or 0 in X.shape or 0 in Z.shape or X.shape[1]!=Z.shape[1]:
        raise ValueError("X和Z必须是特征数相同的非空二维数组")
    return np.mean(X[:,None,:] != Z[None,:,:], axis=2)


def within_cluster_sse(X: np.ndarray, labels: np.ndarray) -> float:
    _numeric_matrix(X,"X"); _labels(labels,X.shape[0],"labels")
    total=0.0
    for label in np.unique(labels):
        cluster=X[labels==label].astype(float); center=np.mean(cluster,axis=0)
        total += float(np.sum((cluster-center)**2))
    return total


def silhouette_samples(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    _numeric_matrix(X,"X"); _labels(labels,X.shape[0],"labels")
    unique=np.unique(labels)
    if unique.size < 2: raise ValueError("轮廓系数至少需要两个簇")
    distances=pairwise_minkowski(X,X,p=2); result=np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        same=np.flatnonzero(labels==labels[i]); same=same[same!=i]
        if same.size==0: continue
        a=float(np.mean(distances[i,same])); b=min(float(np.mean(distances[i,labels==other])) for other in unique if other!=labels[i])
        denominator=max(a,b); result[i]=0.0 if denominator==0 else (b-a)/denominator
    return result


def pair_confusion_counts(labels_true: np.ndarray, labels_pred: np.ndarray) -> dict[str,int]:
    _labels(labels_true,labels_true.size,"labels_true"); _labels(labels_pred,labels_true.size,"labels_pred")
    same_true=labels_true[:,None]==labels_true[None,:]; same_pred=labels_pred[:,None]==labels_pred[None,:]
    upper=np.triu(np.ones((labels_true.size,labels_true.size),dtype=bool),k=1)
    return {"same_same":int(np.count_nonzero(upper&same_true&same_pred)), "same_diff":int(np.count_nonzero(upper&same_true&~same_pred)),
            "diff_same":int(np.count_nonzero(upper&~same_true&same_pred)), "diff_diff":int(np.count_nonzero(upper&~same_true&~same_pred))}


def rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    counts=pair_confusion_counts(labels_true,labels_pred); total=sum(counts.values())
    return 1.0 if total==0 else (counts["same_same"]+counts["diff_diff"])/total


def _comb2(values: np.ndarray) -> np.ndarray:
    return values*(values-1)/2


def adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    _labels(labels_true,labels_true.size,"labels_true"); _labels(labels_pred,labels_true.size,"labels_pred")
    _,true_inv=np.unique(labels_true,return_inverse=True); _,pred_inv=np.unique(labels_pred,return_inverse=True)
    table=np.zeros((true_inv.max()+1,pred_inv.max()+1),dtype=int)
    np.add.at(table,(true_inv,pred_inv),1)
    index=float(np.sum(_comb2(table))); row=float(np.sum(_comb2(table.sum(axis=1)))); col=float(np.sum(_comb2(table.sum(axis=0))))
    total=float(_comb2(np.array([labels_true.size]))[0])
    if total==0: return 1.0
    expected=row*col/total; maximum=0.5*(row+col); denominator=maximum-expected
    return 1.0 if abs(denominator)<1e-15 else (index-expected)/denominator
