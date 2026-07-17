"""参考实现：级联相关网络的确定性结构增长实验。"""
from numbers import Real
import numpy as np

def _matrix(X,name):
    if not isinstance(X,np.ndarray) or X.ndim!=2 or 0 in X.shape or not np.issubdtype(X.dtype,np.number) or not np.all(np.isfinite(X)): raise ValueError(f"{name}必须是非空有限数值二维数组")
def _positive_int(value,name,allow_zero=False):
    lower=0 if allow_zero else 1
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<lower: raise ValueError(f"{name}必须是{'非负' if allow_zero else '正'}整数")
    return int(value)
def _nonnegative(value,name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real) or not np.isfinite(value) or value<0: raise ValueError(f"{name}必须是有限非负数")
    return float(value)
def absolute_correlation(activations,residual):
    _matrix(activations,"activations")
    if not isinstance(residual,np.ndarray) or residual.shape!=(activations.shape[0],) or not np.issubdtype(residual.dtype,np.number) or not np.all(np.isfinite(residual)): raise ValueError("residual必须是形状(n,)的有限数值数组")
    A=activations.astype(float)-activations.mean(axis=0); r=residual.astype(float)-residual.mean(); denominator=np.sqrt(np.sum(A*A,axis=0)*np.sum(r*r)); numerator=np.abs(A.T@r)
    return np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator>0)
def _fit_output(features,y,ridge):
    design=np.column_stack((features,np.ones(features.shape[0]))); penalty=np.eye(design.shape[1]); penalty[-1,-1]=0
    parameters=np.linalg.pinv(design.T@design+ridge*penalty)@design.T@y
    return parameters[:-1],float(parameters[-1])
def fit_cascade_correlation(X,y,*,n_hidden=3,n_candidates=50,random_state=0,ridge=0.0):
    _matrix(X,"X")
    if not isinstance(y,np.ndarray) or y.shape!=(X.shape[0],) or not np.issubdtype(y.dtype,np.number) or not np.all(np.isfinite(y)): raise ValueError("y必须是形状(n,)的有限数值数组")
    n_hidden=_positive_int(n_hidden,"n_hidden",True); n_candidates=_positive_int(n_candidates,"n_candidates"); ridge=_nonnegative(ridge,"ridge")
    if isinstance(random_state,(bool,np.bool_)) or not isinstance(random_state,(int,np.integer)): raise ValueError("random_state必须是整数")
    rng=np.random.default_rng(int(random_state)); features=X.astype(float).copy(); target=y.astype(float); output_weights,output_bias=_fit_output(features,target,ridge); prediction=features@output_weights+output_bias
    history=[float(np.mean((target-prediction)**2))]; hidden=[]; scores=[]
    for _ in range(n_hidden):
        residual=target-prediction; weights=rng.normal(size=(n_candidates,features.shape[1])); biases=rng.normal(size=n_candidates); activations=np.tanh(features@weights.T+biases)
        correlations=absolute_correlation(activations,residual); best=int(np.argmax(correlations)); activation=activations[:,best]
        hidden.append({"weights":weights[best].copy(),"bias":float(biases[best])}); scores.append(float(correlations[best])); features=np.column_stack((features,activation))
        output_weights,output_bias=_fit_output(features,target,ridge); prediction=features@output_weights+output_bias; history.append(float(np.mean((target-prediction)**2)))
    return {"input_dim":X.shape[1],"hidden_units":tuple(hidden),"output_weights":output_weights,"output_bias":output_bias,"mse_history":np.array(history),"selected_correlations":np.array(scores),"ridge":ridge}
def predict_cascade(model,X):
    required={"input_dim","hidden_units","output_weights","output_bias","mse_history","selected_correlations","ridge"}
    if not isinstance(model,dict) or set(model)!=required: raise ValueError("model键集合无效")
    _matrix(X,"X")
    if X.shape[1]!=model["input_dim"]: raise ValueError("X特征数不匹配")
    features=X.astype(float).copy()
    for unit in model["hidden_units"]:
        if not isinstance(unit,dict) or set(unit)!={"weights","bias"} or not isinstance(unit["weights"],np.ndarray) or unit["weights"].shape!=(features.shape[1],) or not np.all(np.isfinite(unit["weights"])) or not np.isfinite(unit["bias"]): raise ValueError("隐藏单元参数无效")
        activation=np.tanh(features@unit["weights"]+float(unit["bias"])); features=np.column_stack((features,activation))
    weights=model["output_weights"]
    if not isinstance(weights,np.ndarray) or weights.shape!=(features.shape[1],) or not np.all(np.isfinite(weights)) or not np.isfinite(model["output_bias"]): raise ValueError("输出层参数无效")
    return features@weights+float(model["output_bias"])
