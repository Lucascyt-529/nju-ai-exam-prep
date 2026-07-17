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
def _responsibilities(responsibilities,n):
    if not isinstance(responsibilities,np.ndarray) or responsibilities.shape!=(n,2) or not np.issubdtype(responsibilities.dtype,np.number) or not np.all(np.isfinite(responsibilities)) or np.any(responsibilities<0) or not np.allclose(responsibilities.sum(axis=1),1): raise ValueError("responsibilities必须是逐行和为1的(n,2)非负数组")
    return responsibilities.astype(float,copy=False)
def expected_complete_log_likelihood(heads,tosses,responsibilities,mixing,probabilities):
    _data(heads,tosses); _parameters(mixing,probabilities); r=_responsibilities(responsibilities,heads.size)
    return float(np.sum(r*_component_logs(heads,tosses,mixing,probabilities,True)))
def responsibility_entropy(responsibilities):
    if not isinstance(responsibilities,np.ndarray) or responsibilities.ndim!=2 or responsibilities.shape[1]!=2 or responsibilities.shape[0]==0: raise ValueError("responsibilities必须是非空(n,2)数组")
    r=_responsibilities(responsibilities,responsibilities.shape[0]); positive=r>0
    return float(-np.sum(r[positive]*np.log(r[positive])))
def evidence_lower_bound(heads,tosses,responsibilities,mixing,probabilities):
    return expected_complete_log_likelihood(heads,tosses,responsibilities,mixing,probabilities)+responsibility_entropy(responsibilities)
def posterior_kl_gap(heads,tosses,responsibilities,mixing,probabilities):
    _data(heads,tosses); _parameters(mixing,probabilities); r=_responsibilities(responsibilities,heads.size)
    posterior=expectation_step(heads,tosses,mixing,probabilities); positive=r>0
    return float(np.sum(r[positive]*(np.log(r[positive])-np.log(posterior[positive]))))
def em_step_diagnostics(heads,tosses,mixing,probabilities):
    _data(heads,tosses); _parameters(mixing,probabilities)
    old_mixing=mixing.astype(float,copy=True); old_probabilities=probabilities.astype(float,copy=True)
    responsibilities=expectation_step(heads,tosses,old_mixing,old_probabilities)
    new_mixing,new_probabilities=maximization_step(heads,tosses,responsibilities)
    old_likelihood=observed_log_likelihood(heads,tosses,old_mixing,old_probabilities)
    new_likelihood=observed_log_likelihood(heads,tosses,new_mixing,new_probabilities)
    old_bound=evidence_lower_bound(heads,tosses,responsibilities,old_mixing,old_probabilities)
    after_m_bound=evidence_lower_bound(heads,tosses,responsibilities,new_mixing,new_probabilities)
    new_responsibilities=expectation_step(heads,tosses,new_mixing,new_probabilities)
    tight_new_bound=evidence_lower_bound(heads,tosses,new_responsibilities,new_mixing,new_probabilities)
    return {"responsibilities":responsibilities,"new_responsibilities":new_responsibilities,"new_mixing":new_mixing,"new_probabilities":new_probabilities,"old_likelihood":old_likelihood,"new_likelihood":new_likelihood,"old_bound":old_bound,"after_m_bound":after_m_bound,"tight_new_bound":tight_new_bound}
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
