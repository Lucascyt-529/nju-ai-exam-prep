"""参考实现：完整协方差高斯混合聚类EM。"""
from numbers import Real
import numpy as np
def _matrix(X,name):
    if not isinstance(X,np.ndarray) or X.ndim!=2 or 0 in X.shape or not np.issubdtype(X.dtype,np.number) or not np.all(np.isfinite(X)): raise ValueError(f"{name}必须是非空有限数值二维数组")
def _parameters(weights,means,covariances,d):
    if not isinstance(weights,np.ndarray) or weights.ndim!=1 or weights.size==0 or np.any(weights<=0) or not np.all(np.isfinite(weights)) or not np.isclose(weights.sum(),1): raise ValueError("weights必须是和为1的正概率")
    if not isinstance(means,np.ndarray) or means.shape!=(weights.size,d) or not np.all(np.isfinite(means)): raise ValueError("means形状无效")
    if not isinstance(covariances,np.ndarray) or covariances.shape!=(weights.size,d,d) or not np.all(np.isfinite(covariances)): raise ValueError("covariances形状无效")
    for covariance in covariances:
        if not np.allclose(covariance,covariance.T,atol=1e-10) or np.min(np.linalg.eigvalsh(covariance))<=0: raise ValueError("每个协方差必须对称正定")
def gaussian_log_density(X,means,covariances):
    _matrix(X,"X")
    if not isinstance(means,np.ndarray) or means.ndim!=2 or means.shape[1]!=X.shape[1] or means.shape[0]==0 or not np.all(np.isfinite(means)) or not isinstance(covariances,np.ndarray) or covariances.shape!=(means.shape[0],X.shape[1],X.shape[1]): raise ValueError("means或covariances形状无效")
    result=np.empty((X.shape[0],means.shape[0])); constant=X.shape[1]*np.log(2*np.pi)
    for k in range(means.shape[0]):
        covariance=covariances[k]
        if not np.all(np.isfinite(covariance)) or not np.allclose(covariance,covariance.T,atol=1e-10): raise ValueError("协方差必须有限对称")
        try: chol=np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as error: raise ValueError("协方差必须正定") from error
        difference=X.astype(float)-means[k]; solved=np.linalg.solve(chol,difference.T).T; result[:,k]=-.5*(constant+2*np.sum(np.log(np.diag(chol)))+np.sum(solved*solved,axis=1))
    return result
def expectation_step(X,weights,means,covariances):
    _matrix(X,"X"); _parameters(weights,means,covariances,X.shape[1]); joint=gaussian_log_density(X,means,covariances)+np.log(weights)[None,:]; maximum=np.max(joint,axis=1,keepdims=True); sums=np.sum(np.exp(joint-maximum),axis=1,keepdims=True); log_norm=maximum+np.log(sums); return np.exp(joint-log_norm),float(np.sum(log_norm))
def maximization_step(X,responsibilities,*,covariance_regularization=1e-6):
    _matrix(X,"X")
    if not isinstance(responsibilities,np.ndarray) or responsibilities.ndim!=2 or responsibilities.shape[0]!=X.shape[0] or responsibilities.shape[1]==0 or np.any(responsibilities<0) or not np.all(np.isfinite(responsibilities)) or not np.allclose(responsibilities.sum(axis=1),1): raise ValueError("responsibilities必须是逐行和为1的非负二维数组")
    if isinstance(covariance_regularization,(bool,np.bool_)) or not isinstance(covariance_regularization,Real) or not np.isfinite(covariance_regularization) or covariance_regularization<=0: raise ValueError("covariance_regularization必须是有限正数")
    counts=responsibilities.sum(axis=0)
    if np.any(counts<=1e-12): raise ValueError("成分软计数过小")
    weights=counts/X.shape[0]; means=responsibilities.T@X/counts[:,None]; covariances=np.empty((len(counts),X.shape[1],X.shape[1]))
    for k in range(len(counts)):
        difference=X-means[k]; covariances[k]=(difference.T*responsibilities[:,k])@difference/counts[k]+float(covariance_regularization)*np.eye(X.shape[1])
    return weights,means,covariances
def _initialize(X,k,rng,regularization):
    indices=[int(rng.integers(X.shape[0]))]; nearest=np.sum((X-X[indices[0]])**2,axis=1)
    while len(indices)<k:
        available=np.ones(X.shape[0],dtype=bool); available[indices]=False; probabilities=np.where(available,nearest,0); total=probabilities.sum(); index=int(rng.choice(X.shape[0],p=probabilities/total)) if total>0 else int(np.flatnonzero(available)[0]); indices.append(index); nearest=np.minimum(nearest,np.sum((X-X[index])**2,axis=1))
    means=X[indices].astype(float).copy()
    for _ in range(20):
        distances=np.sum((X[:,None,:]-means[None,:,:])**2,axis=2); labels=np.argmin(distances,axis=1); updated=means.copy()
        for component in range(k):
            if np.any(labels==component): updated[component]=X[labels==component].mean(axis=0)
        if np.allclose(updated,means): break
        means=updated
    distances=np.sum((X[:,None,:]-means[None,:,:])**2,axis=2); labels=np.argmin(distances,axis=1); counts=np.bincount(labels,minlength=k).astype(float)
    global_centered=X-X.mean(axis=0); global_covariance=global_centered.T@global_centered/max(X.shape[0]-1,1)+regularization*np.eye(X.shape[1]); covariances=[]
    for component in range(k):
        cluster=X[labels==component]
        if len(cluster)<=1: covariance=global_covariance.copy()
        else:
            centered=cluster-means[component]; covariance=centered.T@centered/len(cluster)+regularization*np.eye(X.shape[1])
        covariances.append(covariance)
    return counts/counts.sum(),means,np.asarray(covariances)
def fit_gaussian_mixture(X,n_components,*,random_state=0,max_iterations=200,tolerance=1e-8,covariance_regularization=1e-6):
    _matrix(X,"X")
    if isinstance(n_components,(bool,np.bool_)) or not isinstance(n_components,(int,np.integer)) or n_components<=0 or n_components>X.shape[0]: raise ValueError("n_components必须是1到样本数之间的整数")
    if isinstance(random_state,(bool,np.bool_)) or not isinstance(random_state,(int,np.integer)) or isinstance(max_iterations,(bool,np.bool_)) or not isinstance(max_iterations,(int,np.integer)) or max_iterations<=0: raise ValueError("随机种子或迭代次数无效")
    if not isinstance(tolerance,Real) or isinstance(tolerance,(bool,np.bool_)) or not np.isfinite(tolerance) or tolerance<=0 or not isinstance(covariance_regularization,Real) or isinstance(covariance_regularization,(bool,np.bool_)) or not np.isfinite(covariance_regularization) or covariance_regularization<=0: raise ValueError("容差和协方差正则必须是有限正数")
    weights,means,covariances=_initialize(X.astype(float),int(n_components),np.random.default_rng(int(random_state)),float(covariance_regularization)); responsibilities,likelihood=expectation_step(X,weights,means,covariances); history=[likelihood]
    for _ in range(int(max_iterations)):
        weights,means,covariances=maximization_step(X,responsibilities,covariance_regularization=float(covariance_regularization)); responsibilities,likelihood=expectation_step(X,weights,means,covariances); history.append(likelihood)
        if history[-1]-history[-2]<=float(tolerance): break
    order=np.lexsort(means[:,::-1].T); weights,means,covariances,responsibilities=weights[order],means[order],covariances[order],responsibilities[:,order]
    return {"weights":weights,"means":means,"covariances":covariances,"responsibilities":responsibilities,"labels":np.argmax(responsibilities,axis=1),"log_likelihood_history":np.asarray(history),"iterations":len(history)-1}
