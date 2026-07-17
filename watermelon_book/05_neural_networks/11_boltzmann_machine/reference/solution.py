"""参考实现：小型全可见Boltzmann机的精确分布与Gibbs采样。"""
from numbers import Real
import numpy as np
def _positive(value,name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real) or not np.isfinite(value) or value<=0: raise ValueError(f"{name}必须是有限正数")
    return float(value)
def _parameters(weights,biases):
    if not isinstance(weights,np.ndarray) or weights.ndim!=2 or weights.shape[0]!=weights.shape[1] or weights.shape[0]==0 or not np.issubdtype(weights.dtype,np.number) or not np.all(np.isfinite(weights)) or not np.allclose(weights,weights.T,atol=1e-12) or not np.allclose(np.diag(weights),0,atol=1e-12): raise ValueError("weights必须是有限、对称、零对角方阵")
    if not isinstance(biases,np.ndarray) or biases.shape!=(weights.shape[0],) or not np.issubdtype(biases.dtype,np.number) or not np.all(np.isfinite(biases)): raise ValueError("biases必须是匹配单元数的有限一维数组")
def _binary(value,shape,name):
    if not isinstance(value,np.ndarray) or value.shape!=shape or not np.all(np.isin(value,[0,1])): raise ValueError(f"{name}必须是形状{shape}的二值数组")
def enumerate_binary_states(n_units):
    if isinstance(n_units,(bool,np.bool_)) or not isinstance(n_units,(int,np.integer)) or n_units<=0 or n_units>20: raise ValueError("n_units必须是1到20之间的整数")
    numbers=np.arange(2**int(n_units),dtype=np.uint64); shifts=np.arange(int(n_units)-1,-1,-1,dtype=np.uint64)
    return ((numbers[:,None]>>shifts)&1).astype(int)
def energy(states,weights,biases):
    _parameters(weights,biases)
    if not isinstance(states,np.ndarray) or states.ndim!=2 or states.shape[1]!=weights.shape[0] or states.shape[0]==0 or not np.all(np.isin(states,[0,1])): raise ValueError("states必须是非空(n,d)二值数组")
    S=states.astype(float); return -.5*np.einsum("ni,ij,nj->n",S,weights.astype(float),S)-S@biases.astype(float)
def exact_distribution(weights,biases,*,temperature=1.0):
    _parameters(weights,biases); temperature=_positive(temperature,"temperature"); states=enumerate_binary_states(weights.shape[0]); energies=energy(states,weights,biases); logits=-energies/temperature; shifted=logits-np.max(logits); unnormalized=np.exp(shifted); probabilities=unnormalized/unnormalized.sum(); S=states.astype(float)
    mean=probabilities@S; second=S.T@(S*probabilities[:,None])
    return {"states":states,"energies":energies,"probabilities":probabilities,"mean":mean,"second_moment":second,"log_partition":np.array(np.max(logits)+np.log(unnormalized.sum()))}
def conditional_probability_one(state,unit,weights,biases,*,temperature=1.0):
    _parameters(weights,biases); _binary(state,(weights.shape[0],),"state"); temperature=_positive(temperature,"temperature")
    if isinstance(unit,(bool,np.bool_)) or not isinstance(unit,(int,np.integer)) or unit<0 or unit>=len(state): raise ValueError("unit下标无效")
    field=float(biases[unit]+weights[unit]@state); z=field/temperature
    return float(1/(1+np.exp(-z))) if z>=0 else float(np.exp(z)/(1+np.exp(z)))
def gibbs_sample(initial_state,weights,biases,*,n_sweeps,random_state=0,temperature=1.0):
    _parameters(weights,biases); _binary(initial_state,(weights.shape[0],),"initial_state"); temperature=_positive(temperature,"temperature")
    if isinstance(n_sweeps,(bool,np.bool_)) or not isinstance(n_sweeps,(int,np.integer)) or n_sweeps<0: raise ValueError("n_sweeps必须是非负整数")
    if isinstance(random_state,(bool,np.bool_)) or not isinstance(random_state,(int,np.integer)): raise ValueError("random_state必须是整数")
    rng=np.random.default_rng(int(random_state)); state=initial_state.astype(int).copy(); trajectory=[state.copy()]
    for _ in range(int(n_sweeps)):
        for unit in range(len(state)):
            probability=conditional_probability_one(state,unit,weights,biases,temperature=temperature); state[unit]=int(rng.random()<probability)
        trajectory.append(state.copy())
    return np.asarray(trajectory)
