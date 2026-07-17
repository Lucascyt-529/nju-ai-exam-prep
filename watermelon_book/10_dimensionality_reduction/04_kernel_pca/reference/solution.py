"""参考实现：线性、多项式和RBF核 PCA。"""

from numbers import Real
import numpy as np

KERNELS={"linear","polynomial","rbf"}

def _matrix(X,name):
    if not isinstance(X,np.ndarray) or X.ndim!=2 or 0 in X.shape or not np.issubdtype(X.dtype,np.number) or not np.all(np.isfinite(X)): raise ValueError(f"{name}必须是非空有限数值二维数组")

def _config(d,kernel,degree,gamma,coef0):
    if kernel not in KERNELS: raise ValueError("kernel必须是linear、polynomial或rbf")
    if isinstance(degree,(bool,np.bool_)) or not isinstance(degree,(int,np.integer)) or degree<=0: raise ValueError("degree必须是正整数")
    gamma=1.0/d if gamma is None else gamma
    if isinstance(gamma,(bool,np.bool_)) or not isinstance(gamma,Real) or not np.isfinite(gamma) or gamma<=0: raise ValueError("gamma必须是有限正数")
    if isinstance(coef0,(bool,np.bool_)) or not isinstance(coef0,Real) or not np.isfinite(coef0): raise ValueError("coef0必须是有限数值")
    return str(kernel),int(degree),float(gamma),float(coef0)

def kernel_matrix(X,Z,*,kernel="rbf",degree=3,gamma=None,coef0=1.0):
    _matrix(X,"X"); _matrix(Z,"Z")
    if X.shape[1]!=Z.shape[1]: raise ValueError("X和Z特征数必须一致")
    kind,degree,gamma,coef0=_config(X.shape[1],kernel,degree,gamma,coef0)
    Xf,Zf=X.astype(float),Z.astype(float); dot=Xf@Zf.T
    if kind=="linear": result=dot
    elif kind=="polynomial": result=(gamma*dot+coef0)**degree
    else:
        squared=np.sum(Xf*Xf,axis=1)[:,None]+np.sum(Zf*Zf,axis=1)[None,:]-2*dot
        result=np.exp(-gamma*np.maximum(squared,0))
    if not np.all(np.isfinite(result)): raise ValueError("核矩阵包含非有限值")
    return result

def center_train_kernel(K):
    _matrix(K,"K")
    if K.shape[0]!=K.shape[1] or not np.allclose(K,K.T,atol=1e-12,rtol=1e-12): raise ValueError("训练核矩阵必须是对称方阵")
    column_mean=K.astype(float).mean(axis=0); grand_mean=float(column_mean.mean())
    centered=K-column_mean[None,:]-column_mean[:,None]+grand_mean
    return 0.5*(centered+centered.T),column_mean,grand_mean

def _orient(V):
    result=V.copy()
    for j in range(result.shape[1]):
        pivot=int(np.argmax(np.abs(result[:,j])))
        if result[pivot,j]<0: result[:,j]*=-1
    return result

def fit_kernel_pca(X,n_components,*,kernel="rbf",degree=3,gamma=None,coef0=1.0,eigen_tolerance=1e-12):
    _matrix(X,"X")
    if isinstance(n_components,(bool,np.bool_)) or not isinstance(n_components,(int,np.integer)) or n_components<=0 or n_components>X.shape[0]: raise ValueError("n_components必须是1到样本数之间的整数")
    if isinstance(eigen_tolerance,(bool,np.bool_)) or not isinstance(eigen_tolerance,Real) or not np.isfinite(eigen_tolerance) or eigen_tolerance<=0: raise ValueError("eigen_tolerance必须是有限正数")
    kind,degree,gamma,coef0=_config(X.shape[1],kernel,degree,gamma,coef0)
    K=kernel_matrix(X,X,kernel=kind,degree=degree,gamma=gamma,coef0=coef0); Kc,column_mean,grand_mean=center_train_kernel(K)
    values,vectors=np.linalg.eigh(Kc); order=np.argsort(values)[::-1]; values=values[order]; vectors=_orient(vectors[:,order])
    positive=np.flatnonzero(values>float(eigen_tolerance))
    if positive.size<int(n_components): raise ValueError("正特征值数量不足，无法构造指定维度")
    values=values[:int(n_components)]; vectors=vectors[:,:int(n_components)]; coordinates=vectors*np.sqrt(values)[None,:]
    return {"X_train":X.astype(float).copy(),"kernel":kind,"degree":degree,"gamma":gamma,"coef0":coef0,"column_mean":column_mean,"grand_mean":grand_mean,"eigenvalues":values,"eigenvectors":vectors,"coordinates":coordinates,"centered_kernel":Kc}

def transform_kernel_pca(model,X):
    required={"X_train","kernel","degree","gamma","coef0","column_mean","grand_mean","eigenvalues","eigenvectors","coordinates","centered_kernel"}
    if not isinstance(model,dict) or set(model)!=required: raise ValueError("model键集合无效")
    _matrix(X,"X"); _matrix(model["X_train"],"model X_train")
    if X.shape[1]!=model["X_train"].shape[1]: raise ValueError("X特征数不匹配")
    values=model["eigenvalues"]; vectors=model["eigenvectors"]; column_mean=model["column_mean"]
    if not isinstance(values,np.ndarray) or values.ndim!=1 or np.any(values<=0) or not np.all(np.isfinite(values)) or not isinstance(vectors,np.ndarray) or vectors.shape!=(model["X_train"].shape[0],values.size) or not isinstance(column_mean,np.ndarray) or column_mean.shape!=(model["X_train"].shape[0],): raise ValueError("model谱参数无效")
    K=kernel_matrix(X,model["X_train"],kernel=model["kernel"],degree=model["degree"],gamma=model["gamma"],coef0=model["coef0"])
    centered=K-column_mean[None,:]-K.mean(axis=1)[:,None]+float(model["grand_mean"])
    return centered@vectors/np.sqrt(values)[None,:]
