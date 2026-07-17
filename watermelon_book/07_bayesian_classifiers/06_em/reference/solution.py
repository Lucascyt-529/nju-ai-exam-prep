"""参考实现：二项混合隐变量模型的稳定EM。"""
from math import lgamma
from numbers import Real
import numpy as np
def _data(heads,tosses):
    if not isinstance(heads,np.ndarray) or not isinstance(tosses,np.ndarray) or heads.ndim!=1 or heads.shape!=tosses.shape or heads.size==0 or not np.issubdtype(heads.dtype,np.integer) or not np.issubdtype(tosses.dtype,np.integer) or np.any(tosses<=0) or np.any(heads<0) or np.any(heads>tosses): raise ValueError("heads和tosses必须是合法同形整数计数")
def _parameters(mixing,probabilities):
    if not isinstance(mixing,np.ndarray) or mixing.shape!=(2,) or not np.all(np.isfinite(mixing)) or np.any(mixing<=0) or not np.isclose(mixing.sum(),1): raise ValueError("mixing必须是和为1的两个正概率")
    if not isinstance(probabilities,np.ndarray) or probabilities.shape!=(2,) or not np.all(np.isfinite(probabilities)) or np.any(probabilities<=0) or np.any(probabilities>=1): raise ValueError("probabilities必须是两个开区间(0,1)概率")
def _component_logs(heads,tosses,mixing,probabilities,include_combination=False):
    failures=tosses-heads; logs=np.log(mixing)[None,:]+heads[:,None]*np.log(probabilities)[None,:]+failures[:,None]*np.log1p(-probabilities)[None,:]
    if include_combination:
        combination=np.array([lgamma(int(n)+1)-lgamma(int(h)+1)-lgamma(int(n-h)+1) for h,n in zip(heads,tosses)])
        logs=logs+combination[:,None]
    return logs
def _logsumexp_rows(values):
    maximum=np.max(values,axis=1,keepdims=True); return maximum[:,0]+np.log(np.sum(np.exp(values-maximum),axis=1))
def expectation_step(heads,tosses,mixing,probabilities):
    _data(heads,tosses); _parameters(mixing,probabilities); logs=_component_logs(heads,tosses,mixing,probabilities); normalizer=_logsumexp_rows(logs)
    return np.exp(logs-normalizer[:,None])
def maximization_step(heads,tosses,responsibilities):
    _data(heads,tosses)
    if not isinstance(responsibilities,np.ndarray) or responsibilities.shape!=(heads.size,2) or not np.all(np.isfinite(responsibilities)) or np.any(responsibilities<0) or not np.allclose(responsibilities.sum(axis=1),1): raise ValueError("responsibilities必须是逐行和为1的(n,2)非负数组")
    weights=responsibilities.sum(axis=0)
    if np.any(weights<=0): raise ValueError("每个成分必须具有正软计数")
    mixing=weights/heads.size; probabilities=(responsibilities.T@heads)/(responsibilities.T@tosses)
    probabilities=np.clip(probabilities,1e-12,1-1e-12); return mixing,probabilities
def observed_log_likelihood(heads,tosses,mixing,probabilities):
    _data(heads,tosses); _parameters(mixing,probabilities); return float(np.sum(_logsumexp_rows(_component_logs(heads,tosses,mixing,probabilities,True))))
def fit_coin_mixture_em(heads,tosses,*,initial_mixing=None,initial_probabilities=None,max_iterations=200,tolerance=1e-9):
    _data(heads,tosses); mixing=np.array([.5,.5]) if initial_mixing is None else np.asarray(initial_mixing,dtype=float).copy(); probabilities=np.array([.3,.7]) if initial_probabilities is None else np.asarray(initial_probabilities,dtype=float).copy(); _parameters(mixing,probabilities)
    if isinstance(max_iterations,(bool,np.bool_)) or not isinstance(max_iterations,(int,np.integer)) or max_iterations<=0: raise ValueError("max_iterations必须是正整数")
    if isinstance(tolerance,(bool,np.bool_)) or not isinstance(tolerance,Real) or not np.isfinite(tolerance) or tolerance<=0: raise ValueError("tolerance必须是有限正数")
    history=[observed_log_likelihood(heads,tosses,mixing,probabilities)]
    for _ in range(int(max_iterations)):
        responsibilities=expectation_step(heads,tosses,mixing,probabilities); mixing,probabilities=maximization_step(heads,tosses,responsibilities); history.append(observed_log_likelihood(heads,tosses,mixing,probabilities))
        if history[-1]-history[-2]<=float(tolerance): break
    responsibilities=expectation_step(heads,tosses,mixing,probabilities)
    return {"mixing":mixing,"probabilities":probabilities,"responsibilities":responsibilities,"log_likelihood_history":np.asarray(history),"iterations":len(history)-1}
