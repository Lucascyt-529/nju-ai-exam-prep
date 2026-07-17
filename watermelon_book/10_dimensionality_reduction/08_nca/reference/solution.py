"""参考实现：方阵线性变换形式的NCA。"""
from numbers import Real
import numpy as np
def _matrix(X,name):
    if not isinstance(X,np.ndarray) or X.ndim!=2 or 0 in X.shape or not np.issubdtype(X.dtype,np.number) or not np.all(np.isfinite(X)): raise ValueError(f"{name}必须是非空有限数值二维数组")
def _inputs(X,y,A):
    _matrix(X,"X"); _matrix(A,"transformation")
    if A.shape!=(X.shape[1],X.shape[1]): raise ValueError("transformation必须是(d,d)方阵")
    if not isinstance(y,np.ndarray) or y.shape!=(X.shape[0],) or (y.dtype.kind in "fc" and not np.all(np.isfinite(y))) or np.unique(y).size<2: raise ValueError("y必须是含至少两类的(n,)标签")
    if X.shape[0]<2: raise ValueError("NCA至少需要两个样本")
def neighbor_probabilities(X,transformation):
    _matrix(X,"X"); _matrix(transformation,"transformation")
    if transformation.shape!=(X.shape[1],X.shape[1]) or X.shape[0]<2: raise ValueError("transformation形状或样本数无效")
    transformed=X.astype(float)@transformation.astype(float).T; difference=transformed[:,None,:]-transformed[None,:,:]; squared=np.sum(difference*difference,axis=2); logits=-squared; np.fill_diagonal(logits,-np.inf); maximum=np.max(logits,axis=1,keepdims=True); exponentials=np.exp(logits-maximum); np.fill_diagonal(exponentials,0); return exponentials/exponentials.sum(axis=1,keepdims=True)
def _same_mask(y):
    mask=y[:,None]==y[None,:]; np.fill_diagonal(mask,False); return mask
def nca_objective(X,y,transformation):
    _inputs(X,y,transformation); probabilities=neighbor_probabilities(X,transformation); return float(np.mean(np.sum(probabilities*_same_mask(y),axis=1)))
def nca_gradient(X,y,transformation):
    _inputs(X,y,transformation); probabilities=neighbor_probabilities(X,transformation); same=_same_mask(y); correct=np.sum(probabilities*same,axis=1); differences=X.astype(float)[:,None,:]-X.astype(float)[None,:,:]; outer=np.einsum("ijd,ije->ijde",differences,differences); expected=np.einsum("ij,ijde->ide",probabilities,outer); correct_expected=np.einsum("ij,ij,ijde->ide",probabilities,same,outer); bracket=np.sum(correct[:,None,None]*expected-correct_expected,axis=0); return 2*transformation.astype(float)@bracket/X.shape[0]
def fit_nca(X,y,*,learning_rate=.1,max_iterations=200,tolerance=1e-9):
    _matrix(X,"X"); A=np.eye(X.shape[1]); _inputs(X,y,A)
    if isinstance(learning_rate,(bool,np.bool_)) or not isinstance(learning_rate,Real) or not np.isfinite(learning_rate) or learning_rate<=0 or isinstance(tolerance,(bool,np.bool_)) or not isinstance(tolerance,Real) or not np.isfinite(tolerance) or tolerance<=0: raise ValueError("learning_rate和tolerance必须是有限正数")
    if isinstance(max_iterations,(bool,np.bool_)) or not isinstance(max_iterations,(int,np.integer)) or max_iterations<=0: raise ValueError("max_iterations必须是正整数")
    current=nca_objective(X,y,A); history=[current]
    for _ in range(int(max_iterations)):
        gradient=nca_gradient(X,y,A); step=float(learning_rate); accepted=False
        for _ in range(30):
            candidate=A+step*gradient; value=nca_objective(X,y,candidate)
            if value>=current-1e-12: accepted=True; break
            step*=.5
        if not accepted: break
        A=candidate; history.append(value)
        if value-current<=float(tolerance): current=value; break
        current=value
    probabilities=neighbor_probabilities(X,A); return {"transformation":A,"metric":A.T@A,"probabilities":probabilities,"correct_probabilities":np.sum(probabilities*_same_mask(y),axis=1),"objective_history":np.asarray(history),"iterations":len(history)-1}
