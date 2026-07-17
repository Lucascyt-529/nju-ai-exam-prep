"""参考实现：二值RBM条件概率和CD-1训练。"""
from numbers import Real
import numpy as np
def _binary(X,name):
    if not isinstance(X,np.ndarray) or X.ndim!=2 or 0 in X.shape or not np.all(np.isin(X,[0,1])): raise ValueError(f"{name}必须是非空二值二维数组")
def _finite_matrix(X,name):
    if not isinstance(X,np.ndarray) or X.ndim!=2 or 0 in X.shape or not np.issubdtype(X.dtype,np.number) or not np.all(np.isfinite(X)): raise ValueError(f"{name}必须是非空有限数值二维数组")
def _parameters(weights,visible_bias,hidden_bias):
    _finite_matrix(weights,"weights")
    if not isinstance(visible_bias,np.ndarray) or visible_bias.shape!=(weights.shape[0],) or not np.all(np.isfinite(visible_bias)) or not isinstance(hidden_bias,np.ndarray) or hidden_bias.shape!=(weights.shape[1],) or not np.all(np.isfinite(hidden_bias)): raise ValueError("偏置形状或数值无效")
def _sigmoid(values):
    result=np.empty_like(values,dtype=float); positive=values>=0; result[positive]=1/(1+np.exp(-values[positive])); exponential=np.exp(values[~positive]); result[~positive]=exponential/(1+exponential); return result
def hidden_probabilities(visible,weights,hidden_bias):
    _binary(visible,"visible"); _finite_matrix(weights,"weights")
    if visible.shape[1]!=weights.shape[0] or not isinstance(hidden_bias,np.ndarray) or hidden_bias.shape!=(weights.shape[1],) or not np.all(np.isfinite(hidden_bias)): raise ValueError("可见层、权重或隐偏置形状无效")
    return _sigmoid(visible.astype(float)@weights.astype(float)+hidden_bias)
def visible_probabilities(hidden,weights,visible_bias):
    _binary(hidden,"hidden"); _finite_matrix(weights,"weights")
    if hidden.shape[1]!=weights.shape[1] or not isinstance(visible_bias,np.ndarray) or visible_bias.shape!=(weights.shape[0],) or not np.all(np.isfinite(visible_bias)): raise ValueError("隐层、权重或显偏置形状无效")
    return _sigmoid(hidden.astype(float)@weights.astype(float).T+visible_bias)
def _sample(probabilities,rng): return (rng.random(probabilities.shape)<probabilities).astype(int)
def _cd1(visible_data,weights,visible_bias,hidden_bias,learning_rate,rng):
    hidden_probability_0=hidden_probabilities(visible_data,weights,hidden_bias); hidden_state_0=_sample(hidden_probability_0,rng); visible_probability_1=visible_probabilities(hidden_state_0,weights,visible_bias); visible_state_1=_sample(visible_probability_1,rng); hidden_probability_1=hidden_probabilities(visible_state_1,weights,hidden_bias); hidden_state_1=_sample(hidden_probability_1,rng); n=len(visible_data)
    gradients={"weights":visible_data.astype(float).T@hidden_state_0/n-visible_state_1.astype(float).T@hidden_state_1/n,"visible_bias":np.mean(visible_data-visible_state_1,axis=0),"hidden_bias":np.mean(hidden_state_0-hidden_state_1,axis=0)}
    return {"weights":weights+learning_rate*gradients["weights"],"visible_bias":visible_bias+learning_rate*gradients["visible_bias"],"hidden_bias":hidden_bias+learning_rate*gradients["hidden_bias"],"gradients":gradients,"hidden_probability_0":hidden_probability_0,"hidden_state_0":hidden_state_0,"visible_probability_1":visible_probability_1,"visible_state_1":visible_state_1,"hidden_probability_1":hidden_probability_1,"hidden_state_1":hidden_state_1}
def cd1_step(visible_data,weights,visible_bias,hidden_bias,*,learning_rate=.1,random_state=0):
    _binary(visible_data,"visible_data"); _parameters(weights,visible_bias,hidden_bias)
    if visible_data.shape[1]!=weights.shape[0]: raise ValueError("visible_data特征数与weights不匹配")
    if isinstance(learning_rate,(bool,np.bool_)) or not isinstance(learning_rate,Real) or not np.isfinite(learning_rate) or learning_rate<=0 or isinstance(random_state,(bool,np.bool_)) or not isinstance(random_state,(int,np.integer)): raise ValueError("学习率或随机种子无效")
    return _cd1(visible_data,weights.astype(float).copy(),visible_bias.astype(float).copy(),hidden_bias.astype(float).copy(),float(learning_rate),np.random.default_rng(int(random_state)))
def _reconstruction_cross_entropy(V,W,vb,hb):
    hp=hidden_probabilities(V,W,hb); vp=_sigmoid(hp@W.T+vb); clipped=np.clip(vp,1e-12,1-1e-12); return float(-np.mean(V*np.log(clipped)+(1-V)*np.log1p(-clipped)))
def fit_rbm_cd1(visible_data,n_hidden,*,epochs=100,learning_rate=.1,random_state=0):
    _binary(visible_data,"visible_data")
    if isinstance(n_hidden,(bool,np.bool_)) or not isinstance(n_hidden,(int,np.integer)) or n_hidden<=0 or isinstance(epochs,(bool,np.bool_)) or not isinstance(epochs,(int,np.integer)) or epochs<=0 or isinstance(random_state,(bool,np.bool_)) or not isinstance(random_state,(int,np.integer)): raise ValueError("n_hidden、epochs或random_state无效")
    if isinstance(learning_rate,(bool,np.bool_)) or not isinstance(learning_rate,Real) or not np.isfinite(learning_rate) or learning_rate<=0: raise ValueError("learning_rate必须是有限正数")
    rng=np.random.default_rng(int(random_state)); weights=rng.normal(0,.01,size=(visible_data.shape[1],int(n_hidden))); visible_bias=np.zeros(visible_data.shape[1]); hidden_bias=np.zeros(int(n_hidden)); history=[_reconstruction_cross_entropy(visible_data,weights,visible_bias,hidden_bias)]
    for _ in range(int(epochs)):
        step=_cd1(visible_data,weights,visible_bias,hidden_bias,float(learning_rate),rng); weights,visible_bias,hidden_bias=step["weights"],step["visible_bias"],step["hidden_bias"]; history.append(_reconstruction_cross_entropy(visible_data,weights,visible_bias,hidden_bias))
    return {"weights":weights,"visible_bias":visible_bias,"hidden_bias":hidden_bias,"reconstruction_history":np.asarray(history),"epochs":int(epochs)}
