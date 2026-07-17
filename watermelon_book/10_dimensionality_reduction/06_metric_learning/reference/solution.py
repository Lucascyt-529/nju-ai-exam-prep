"""参考实现：PSD Mahalanobis矩阵的成对约束学习。"""
from numbers import Real
import numpy as np
def _matrix(X,name):
    if not isinstance(X,np.ndarray) or X.ndim!=2 or 0 in X.shape or not np.issubdtype(X.dtype,np.number) or not np.all(np.isfinite(X)): raise ValueError(f"{name}必须是非空有限数值二维数组")
def _labels(y,n):
    if not isinstance(y,np.ndarray) or y.shape!=(n,) or (y.dtype.kind in "fc" and not np.all(np.isfinite(y))) or np.unique(y).size<2: raise ValueError("y必须是含至少两类的(n,)标签")
def _number(value,name,positive=True):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real) or not np.isfinite(value) or (positive and value<=0) or (not positive and value<0): raise ValueError(f"{name}数值无效")
    return float(value)
def _metric(M,d):
    _matrix(M,"M")
    if M.shape!=(d,d) or not np.allclose(M,M.T,atol=1e-10): raise ValueError("M必须是匹配特征数的对称方阵")
    if np.min(np.linalg.eigvalsh(M)) < -1e-9: raise ValueError("M必须半正定")
def pairwise_mahalanobis(X,M):
    _matrix(X,"X"); _metric(M,X.shape[1]); diff=X.astype(float)[:,None,:]-X.astype(float)[None,:,:]
    squared=np.einsum("ijd,df,ijf->ij",diff,M.astype(float),diff)
    return np.sqrt(np.maximum(squared,0))
def project_psd(M):
    _matrix(M,"M")
    if M.shape[0]!=M.shape[1]: raise ValueError("M必须是方阵")
    symmetric=(M.astype(float)+M.astype(float).T)/2; values,vectors=np.linalg.eigh(symmetric)
    return (vectors*np.maximum(values,0)[None,:])@vectors.T
def _pairs(X,y):
    left,right=np.triu_indices(X.shape[0],1); diff=X[left].astype(float)-X[right].astype(float); same=y[left]==y[right]
    if not np.any(same) or np.all(same): raise ValueError("数据必须同时产生同类对和异类对")
    return diff,same
def _loss_gradient(X,y,M,margin,different_weight,regularization):
    diff,same=_pairs(X,y); outers=np.einsum("ni,nj->nij",diff,diff); squared=np.einsum("ni,ij,nj->n",diff,M,diff)
    active=(~same)&(squared<margin)
    loss=float(np.sum(squared[same])+different_weight*np.sum(margin-squared[active])+regularization*np.sum((M-np.eye(M.shape[0]))**2))
    gradient=np.sum(outers[same],axis=0)-different_weight*np.sum(outers[active],axis=0)+2*regularization*(M-np.eye(M.shape[0]))
    return loss,gradient
def metric_objective(X,y,M,*,margin=1.,different_weight=1.,regularization=.01):
    _matrix(X,"X"); _labels(y,X.shape[0]); _metric(M,X.shape[1]); margin=_number(margin,"margin"); different_weight=_number(different_weight,"different_weight"); regularization=_number(regularization,"regularization",False)
    return _loss_gradient(X,y,M,margin,different_weight,regularization)[0]
def fit_pair_metric(X,y,*,margin=1.,different_weight=1.,regularization=.01,learning_rate=.05,max_iterations=100,tolerance=1e-9):
    _matrix(X,"X"); _labels(y,X.shape[0]); margin=_number(margin,"margin"); different_weight=_number(different_weight,"different_weight"); regularization=_number(regularization,"regularization",False); learning_rate=_number(learning_rate,"learning_rate"); tolerance=_number(tolerance,"tolerance")
    if isinstance(max_iterations,(bool,np.bool_)) or not isinstance(max_iterations,(int,np.integer)) or max_iterations<=0: raise ValueError("max_iterations必须是正整数")
    M=np.eye(X.shape[1]); current,gradient=_loss_gradient(X,y,M,margin,different_weight,regularization); history=[current]
    for _ in range(int(max_iterations)):
        step=learning_rate; accepted=False
        for _ in range(30):
            candidate=project_psd(M-step*gradient); value,new_gradient=_loss_gradient(X,y,candidate,margin,different_weight,regularization)
            if value<=current+1e-12: accepted=True; break
            step*=.5
        if not accepted: break
        M,gradient=candidate,new_gradient; history.append(value)
        if current-value<=tolerance: current=value; break
        current=value
    return {"metric":M,"objective_history":np.array(history),"iterations":len(history)-1,"margin":margin,"different_weight":different_weight,"regularization":regularization}
