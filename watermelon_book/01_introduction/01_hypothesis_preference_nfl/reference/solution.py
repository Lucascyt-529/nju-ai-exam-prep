"""参考实现：有限二分类世界中的归纳偏好与NFL实验。"""
import numpy as np
def _positive_int(value,name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<=0: raise ValueError(f"{name}必须是正整数")
    return int(value)
def enumerate_binary_hypotheses(n_points):
    n=_positive_int(n_points,"n_points"); numbers=np.arange(2**n,dtype=np.uint64)
    shifts=np.arange(n-1,-1,-1,dtype=np.uint64); return ((numbers[:,None]>>shifts)&1).astype(int)
def _hypotheses(H):
    if not isinstance(H,np.ndarray) or H.ndim!=2 or 0 in H.shape or not np.all(np.isin(H,[0,1])): raise ValueError("hypotheses必须是非空二值二维数组")
def version_space(hypotheses,observed_indices,observed_labels):
    _hypotheses(hypotheses); n=hypotheses.shape[1]
    if not isinstance(observed_indices,np.ndarray) or observed_indices.ndim!=1 or observed_indices.size==0 or not np.issubdtype(observed_indices.dtype,np.integer) or np.any(observed_indices<0) or np.any(observed_indices>=n) or np.unique(observed_indices).size!=observed_indices.size: raise ValueError("observed_indices无效")
    if not isinstance(observed_labels,np.ndarray) or observed_labels.shape!=observed_indices.shape or not np.all(np.isin(observed_labels,[0,1])): raise ValueError("observed_labels无效")
    return hypotheses[np.all(hypotheses[:,observed_indices]==observed_labels,axis=1)].copy()
def transition_count(hypothesis):
    if not isinstance(hypothesis,np.ndarray) or hypothesis.ndim!=1 or hypothesis.size==0 or not np.all(np.isin(hypothesis,[0,1])): raise ValueError("hypothesis必须是非空二值一维数组")
    return int(np.count_nonzero(hypothesis[1:]!=hypothesis[:-1]))
def select_smooth_preference(hypotheses):
    _hypotheses(hypotheses); scores=np.array([transition_count(row) for row in hypotheses]); best=np.flatnonzero(scores==scores.min())
    order=np.lexsort(hypotheses[best,::-1].T); return hypotheses[best[order[0]]].copy()
def nfl_unseen_error_experiment(n_points,train_indices):
    n=_positive_int(n_points,"n_points"); targets=enumerate_binary_hypotheses(n)
    if not isinstance(train_indices,np.ndarray) or train_indices.ndim!=1 or train_indices.size==0 or train_indices.size>=n or not np.issubdtype(train_indices.dtype,np.integer) or np.any(train_indices<0) or np.any(train_indices>=n) or np.unique(train_indices).size!=train_indices.size: raise ValueError("train_indices必须是非空、非全集的合法唯一索引")
    unseen=np.setdiff1d(np.arange(n),train_indices); errors_a=[]; errors_b=[]
    for target in targets:
        prediction_a=np.zeros(n,dtype=int); prediction_b=np.arange(n)%2
        prediction_a[train_indices]=target[train_indices]; prediction_b[train_indices]=target[train_indices]
        errors_a.append(np.mean(prediction_a[unseen]!=target[unseen])); errors_b.append(np.mean(prediction_b[unseen]!=target[unseen]))
    return {"learner_zero":float(np.mean(errors_a)),"learner_parity":float(np.mean(errors_b)),"n_targets":int(len(targets))}
