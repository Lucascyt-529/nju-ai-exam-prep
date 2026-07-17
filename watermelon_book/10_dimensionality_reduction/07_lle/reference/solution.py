"""参考实现：局部线性嵌入LLE。"""
from numbers import Real
import numpy as np
def _matrix(X,name):
    if not isinstance(X,np.ndarray) or X.ndim!=2 or 0 in X.shape or not np.issubdtype(X.dtype,np.number) or not np.all(np.isfinite(X)): raise ValueError(f"{name}必须是非空有限数值二维数组")
def _positive(value,name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real) or not np.isfinite(value) or value<=0: raise ValueError(f"{name}必须是有限正数")
    return float(value)
def nearest_neighbors(X,n_neighbors):
    _matrix(X,"X"); n=X.shape[0]
    if isinstance(n_neighbors,(bool,np.bool_)) or not isinstance(n_neighbors,(int,np.integer)) or n_neighbors<=0 or n_neighbors>=n: raise ValueError("n_neighbors必须是1到n-1之间的整数")
    difference=X.astype(float)[:,None,:]-X.astype(float)[None,:,:]; distances=np.sum(difference*difference,axis=2); np.fill_diagonal(distances,np.inf)
    return np.argsort(distances,axis=1,kind="stable")[:,:int(n_neighbors)]
def reconstruction_weights(X,neighbors,*,regularization=1e-3):
    _matrix(X,"X"); regularization=_positive(regularization,"regularization")
    if not isinstance(neighbors,np.ndarray) or neighbors.ndim!=2 or neighbors.shape[0]!=X.shape[0] or neighbors.shape[1]==0 or not np.issubdtype(neighbors.dtype,np.integer) or np.any(neighbors<0) or np.any(neighbors>=X.shape[0]): raise ValueError("neighbors必须是合法(n,k)整数下标")
    if any(len(np.unique(row))!=len(row) or index in row for index,row in enumerate(neighbors)): raise ValueError("每行邻居必须唯一且不能包含自身")
    W=np.zeros((X.shape[0],X.shape[0]),dtype=float); ones=np.ones(neighbors.shape[1])
    for i,row in enumerate(neighbors):
        difference=X[row].astype(float)-X[i].astype(float); gram=difference@difference.T; trace=float(np.trace(gram)); gram=gram+regularization*(trace if trace>0 else 1.0)*np.eye(len(row)); local=np.linalg.solve(gram,ones); local/=local.sum(); W[i,row]=local
    return W
def _reconstruction_error(X,W): return float(np.sum((X-W@X)**2))
def fit_lle(X,n_neighbors,n_components,*,regularization=1e-3):
    _matrix(X,"X"); n=X.shape[0]
    if isinstance(n_components,(bool,np.bool_)) or not isinstance(n_components,(int,np.integer)) or n_components<=0 or n_components>=n: raise ValueError("n_components必须是1到n-1之间的整数")
    neighbors=nearest_neighbors(X,n_neighbors); W=reconstruction_weights(X,neighbors,regularization=regularization); identity=np.eye(n); embedding_matrix=(identity-W).T@(identity-W); embedding_matrix=(embedding_matrix+embedding_matrix.T)/2
    values,vectors=np.linalg.eigh(embedding_matrix); order=np.argsort(values); values=values[order]; vectors=vectors[:,order]; coordinates=vectors[:,1:int(n_components)+1]*np.sqrt(n)
    coordinates-=coordinates.mean(axis=0); scales=np.sqrt(np.mean(coordinates*coordinates,axis=0)); coordinates/=scales
    for column in range(coordinates.shape[1]):
        pivot=int(np.argmax(np.abs(coordinates[:,column])))
        if coordinates[pivot,column]<0: coordinates[:,column]*=-1
    return {"coordinates":coordinates,"neighbors":neighbors,"weights":W,"eigenvalues":values,"embedding_matrix":embedding_matrix,"high_dimensional_error":_reconstruction_error(X.astype(float),W),"low_dimensional_error":_reconstruction_error(coordinates,W)}
