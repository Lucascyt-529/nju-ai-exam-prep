"""参考实现：LVQ1最近原型更新与衰减训练。"""

import numpy as np


def _matrix(X: np.ndarray,name: str)->None:
    if not isinstance(X,np.ndarray) or X.ndim!=2 or 0 in X.shape or not np.issubdtype(X.dtype,np.number) or not np.all(np.isfinite(X)): raise ValueError(f"{name}必须是非空有限数值二维数组")


def _rate(value: object,name: str)->float:
    if not isinstance(value,(int,float,np.integer,np.floating)) or isinstance(value,(bool,np.bool_)) or not np.isfinite(value) or value<=0 or value>1: raise ValueError(f"{name}必须位于(0,1]")
    return float(value)


def _positive_int(value: object,name: str)->int:
    if not isinstance(value,(int,np.integer)) or isinstance(value,(bool,np.bool_)) or value<=0: raise ValueError(f"{name}必须是正整数")
    return int(value)


def squared_distances(X: np.ndarray,prototypes: np.ndarray)->np.ndarray:
    _matrix(X,"X"); _matrix(prototypes,"prototypes")
    if X.shape[1]!=prototypes.shape[1]: raise ValueError("X和prototypes特征数必须一致")
    difference=X.astype(float)[:,None,:]-prototypes.astype(float)[None,:,:]
    return np.sum(difference*difference,axis=2)


def nearest_prototype(X: np.ndarray,prototypes: np.ndarray)->tuple[np.ndarray,np.ndarray]:
    distances=squared_distances(X,prototypes); winners=np.argmin(distances,axis=1)
    return winners,distances


def lvq_update(prototypes: np.ndarray,prototype_labels: np.ndarray,x: np.ndarray,y: object,learning_rate: float)->tuple[np.ndarray,int,bool]:
    _matrix(prototypes,"prototypes"); eta=_rate(learning_rate,"learning_rate")
    if not isinstance(prototype_labels,np.ndarray) or prototype_labels.shape!=(prototypes.shape[0],): raise ValueError("prototype_labels形状必须为(m,)")
    if not isinstance(x,np.ndarray) or x.shape!=(prototypes.shape[1],) or not np.issubdtype(x.dtype,np.number) or not np.all(np.isfinite(x)): raise ValueError("x必须是形状(d,)的有限数值数组")
    winner=int(nearest_prototype(x.reshape(1,-1),prototypes)[0][0]); correct=bool(prototype_labels[winner]==y)
    updated=prototypes.astype(float).copy(); direction=x.astype(float)-updated[winner]
    updated[winner] += eta*direction if correct else -eta*direction
    return updated,winner,correct


def initialize_prototypes(X: np.ndarray,y: np.ndarray,prototypes_per_class: int,random_state: int)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    _matrix(X,"X")
    if not isinstance(y,np.ndarray) or y.shape!=(X.shape[0],) or np.unique(y).size<2: raise ValueError("y必须是形状(n,)且至少含两个类别")
    count=_positive_int(prototypes_per_class,"prototypes_per_class")
    if not isinstance(random_state,(int,np.integer)) or isinstance(random_state,(bool,np.bool_)): raise ValueError("random_state必须是整数")
    rng=np.random.default_rng(int(random_state)); selected=[]; labels=[]
    for label in np.unique(y):
        candidates=np.flatnonzero(y==label)
        if count>candidates.size: raise ValueError("每类原型数不能超过该类样本数")
        chosen=np.sort(rng.choice(candidates,size=count,replace=False)); selected.extend(chosen.tolist()); labels.extend([label]*count)
    indices=np.array(selected,dtype=int)
    return X[indices].astype(float).copy(),np.asarray(labels,dtype=y.dtype),indices


def fit_lvq(X: np.ndarray,y: np.ndarray,*,prototypes_per_class: int=1,epochs: int=20,learning_rate: float=.2,decay: float=.95,random_state: int=0)->dict[str,object]:
    _matrix(X,"X"); total_epochs=_positive_int(epochs,"epochs"); eta=_rate(learning_rate,"learning_rate"); decay_value=_rate(decay,"decay")
    prototypes,prototype_labels,indices=initialize_prototypes(X,y,prototypes_per_class,random_state); history=[prototypes.copy()]; correct_updates=[]
    for epoch in range(total_epochs):
        correct=0; current_rate=eta*(decay_value**epoch)
        for x_i,y_i in zip(X,y):
            prototypes,_,is_correct=lvq_update(prototypes,prototype_labels,x_i,y_i,current_rate); correct+=int(is_correct)
        history.append(prototypes.copy()); correct_updates.append(correct)
    return {"prototypes":prototypes,"prototype_labels":prototype_labels,"initial_indices":indices,
            "history":np.stack(history),"correct_updates":np.array(correct_updates),"n_features":X.shape[1]}


def predict(model: dict[str,object],X: np.ndarray)->np.ndarray:
    _matrix(X,"X")
    if not isinstance(model,dict) or set(model)!={"prototypes","prototype_labels","initial_indices","history","correct_updates","n_features"} or X.shape[1]!=model["n_features"]: raise ValueError("model无效或特征数不匹配")
    winners,_=nearest_prototype(X,model["prototypes"]); return model["prototype_labels"][winners]
