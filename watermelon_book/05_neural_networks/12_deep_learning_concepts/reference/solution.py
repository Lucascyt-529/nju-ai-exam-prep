"""参考实现：线性逐层预训练与一维卷积权共享。"""
import numpy as np
def _matrix(X,name):
    if not isinstance(X,np.ndarray) or X.ndim!=2 or 0 in X.shape or not np.issubdtype(X.dtype,np.number) or not np.all(np.isfinite(X)): raise ValueError(f"{name}必须是非空有限数值二维数组")
def _positive_int(value,name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<=0: raise ValueError(f"{name}必须是正整数")
    return int(value)
def _orient_rows(rows):
    result=rows.copy()
    for i in range(len(result)):
        pivot=int(np.argmax(np.abs(result[i])))
        if result[i,pivot]<0: result[i]*=-1
    return result
def fit_linear_autoencoder_layer(X,n_hidden):
    _matrix(X,"X"); n_hidden=_positive_int(n_hidden,"n_hidden")
    if n_hidden>X.shape[1]: raise ValueError("n_hidden不能超过输入特征数")
    mean=X.astype(float).mean(axis=0); centered=X.astype(float)-mean; covariance=centered.T@centered/max(X.shape[0]-1,1); values,vectors=np.linalg.eigh(covariance); order=np.argsort(values)[::-1]; components=_orient_rows(vectors[:,order[:n_hidden]].T); code=centered@components.T; reconstructed=code@components+mean
    total=max(float(np.sum(np.maximum(values,0))),1e-30); explained=np.maximum(values[order[:n_hidden]],0)/total
    return {"mean":mean,"components":components,"code":code,"reconstructed":reconstructed,"reconstruction_mse":np.array(np.mean((X-reconstructed)**2)),"explained_variance_ratio":explained}
def greedy_layerwise_pretrain(X,hidden_dimensions):
    _matrix(X,"X")
    if not isinstance(hidden_dimensions,tuple) or not hidden_dimensions: raise ValueError("hidden_dimensions必须是非空整数元组")
    current=X.astype(float).copy(); layers=[]; errors=[]; shapes=[]
    for width in hidden_dimensions:
        layer=fit_linear_autoencoder_layer(current,width); stored={"mean":layer["mean"].copy(),"components":layer["components"].copy()}; layers.append(stored); errors.append(float(layer["reconstruction_mse"])); current=layer["code"]; shapes.append(current.shape)
    return {"layers":tuple(layers),"code":current,"code_shapes":tuple(shapes),"reconstruction_errors":np.asarray(errors)}
def encode_layers(X,layers):
    _matrix(X,"X")
    if not isinstance(layers,tuple) or not layers: raise ValueError("layers必须是非空元组")
    current=X.astype(float).copy()
    for layer in layers:
        if not isinstance(layer,dict) or set(layer)!={"mean","components"} or not isinstance(layer["mean"],np.ndarray) or layer["mean"].shape!=(current.shape[1],) or not isinstance(layer["components"],np.ndarray) or layer["components"].ndim!=2 or layer["components"].shape[1]!=current.shape[1] or not np.all(np.isfinite(layer["mean"])) or not np.all(np.isfinite(layer["components"])): raise ValueError("层参数无效")
        current=(current-layer["mean"])@layer["components"].T
    return current
def shared_convolution_1d(X,kernels):
    _matrix(X,"X"); _matrix(kernels,"kernels")
    if kernels.shape[1]>X.shape[1]: raise ValueError("卷积核宽度不能超过输入长度")
    windows=np.lib.stride_tricks.sliding_window_view(X.astype(float),kernels.shape[1],axis=1); return np.einsum("npk,fk->npf",windows,kernels.astype(float))
def convolution_parameter_counts(input_length,kernel_width,n_filters):
    input_length=_positive_int(input_length,"input_length"); kernel_width=_positive_int(kernel_width,"kernel_width"); n_filters=_positive_int(n_filters,"n_filters")
    if kernel_width>input_length: raise ValueError("kernel_width不能超过input_length")
    shared=kernel_width*n_filters; return {"shared":shared,"locally_connected":(input_length-kernel_width+1)*shared}
