"""参考实现：距离、SSE、轮廓系数、Rand与ARI。"""

from numbers import Real
import numpy as np


def _numeric_matrix(X: np.ndarray, name: str) -> None:
    if not isinstance(X, np.ndarray) or X.ndim != 2 or 0 in X.shape or not np.issubdtype(X.dtype, np.number) or not np.all(np.isfinite(X)):
        raise ValueError(f"{name}必须是非空有限数值二维数组")


def _labels(labels: np.ndarray, n: int, name: str) -> None:
    if not isinstance(labels, np.ndarray) or labels.shape != (n,) or labels.size == 0:
        raise ValueError(f"{name}必须是形状(n,)的一维数组")
    if labels.dtype.kind in "fc" and not np.all(np.isfinite(labels)):
        raise ValueError(f"{name}不能包含非有限值")


def pairwise_minkowski(X: np.ndarray, Z: np.ndarray, *, p: float = 2.0) -> np.ndarray:
    _numeric_matrix(X, "X"); _numeric_matrix(Z, "Z")
    if X.shape[1] != Z.shape[1]: raise ValueError("X和Z特征数必须一致")
    if not isinstance(p, (int,float,np.integer,np.floating)) or isinstance(p,(bool,np.bool_)) or not np.isfinite(p) or p < 1:
        raise ValueError("p必须是不小于1的有限数值")
    return np.sum(np.abs(X.astype(float)[:,None,:]-Z.astype(float)[None,:,:])**float(p),axis=2)**(1.0/float(p))


def pairwise_hamming(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    if not isinstance(X,np.ndarray) or not isinstance(Z,np.ndarray) or X.ndim!=2 or Z.ndim!=2 or 0 in X.shape or 0 in Z.shape or X.shape[1]!=Z.shape[1]:
        raise ValueError("X和Z必须是特征数相同的非空二维数组")
    return np.mean(X[:,None,:] != Z[None,:,:], axis=2)


def within_cluster_sse(X: np.ndarray, labels: np.ndarray) -> float:
    _numeric_matrix(X,"X"); _labels(labels,X.shape[0],"labels")
    total=0.0
    for label in np.unique(labels):
        cluster=X[labels==label].astype(float); center=np.mean(cluster,axis=0)
        total += float(np.sum((cluster-center)**2))
    return total


def silhouette_samples(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    _numeric_matrix(X,"X"); _labels(labels,X.shape[0],"labels")
    unique=np.unique(labels)
    if unique.size < 2: raise ValueError("轮廓系数至少需要两个簇")
    distances=pairwise_minkowski(X,X,p=2); result=np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        same=np.flatnonzero(labels==labels[i]); same=same[same!=i]
        if same.size==0: continue
        a=float(np.mean(distances[i,same])); b=min(float(np.mean(distances[i,labels==other])) for other in unique if other!=labels[i])
        denominator=max(a,b); result[i]=0.0 if denominator==0 else (b-a)/denominator
    return result


def pair_confusion_counts(labels_true: np.ndarray, labels_pred: np.ndarray) -> dict[str,int]:
    _labels(labels_true,labels_true.size,"labels_true"); _labels(labels_pred,labels_true.size,"labels_pred")
    same_true=labels_true[:,None]==labels_true[None,:]; same_pred=labels_pred[:,None]==labels_pred[None,:]
    upper=np.triu(np.ones((labels_true.size,labels_true.size),dtype=bool),k=1)
    return {"same_same":int(np.count_nonzero(upper&same_true&same_pred)), "same_diff":int(np.count_nonzero(upper&same_true&~same_pred)),
            "diff_same":int(np.count_nonzero(upper&~same_true&same_pred)), "diff_diff":int(np.count_nonzero(upper&~same_true&~same_pred))}


def rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    counts=pair_confusion_counts(labels_true,labels_pred); total=sum(counts.values())
    return 1.0 if total==0 else (counts["same_same"]+counts["diff_diff"])/total


def _comb2(values: np.ndarray) -> np.ndarray:
    return values*(values-1)/2


def adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    _labels(labels_true,labels_true.size,"labels_true"); _labels(labels_pred,labels_true.size,"labels_pred")
    _,true_inv=np.unique(labels_true,return_inverse=True); _,pred_inv=np.unique(labels_pred,return_inverse=True)
    table=np.zeros((true_inv.max()+1,pred_inv.max()+1),dtype=int)
    np.add.at(table,(true_inv,pred_inv),1)
    index=float(np.sum(_comb2(table))); row=float(np.sum(_comb2(table.sum(axis=1)))); col=float(np.sum(_comb2(table.sum(axis=0))))
    total=float(_comb2(np.array([labels_true.size]))[0])
    if total==0: return 1.0
    expected=row*col/total; maximum=0.5*(row+col); denominator=maximum-expected
    return 1.0 if abs(denominator)<1e-15 else (index-expected)/denominator


def _categorical_matrix(X: np.ndarray, name: str) -> None:
    if not isinstance(X, np.ndarray) or X.ndim != 2 or 0 in X.shape:
        raise ValueError(f"{name}必须是非空二维数组")


def _distance_power(p: float) -> float:
    if (isinstance(p, (bool, np.bool_)) or not isinstance(p, Real)
            or not np.isfinite(p) or p < 1):
        raise ValueError("p必须是不小于1的有限数值")
    return float(p)


def _feature_weights(weights: np.ndarray | None, n_features: int, name: str) -> np.ndarray:
    if weights is None:
        return np.ones(n_features, dtype=float)
    values = np.asarray(weights, dtype=float)
    if (values.shape != (n_features,) or not np.all(np.isfinite(values))
            or np.any(values < 0) or np.sum(values) <= 0):
        raise ValueError(f"{name}必须是形状匹配、非负有限且至少一个为正的一维数组")
    return values


def pairwise_weighted_minkowski(X: np.ndarray, Z: np.ndarray, *, p: float = 2.0,
                                weights: np.ndarray | None = None) -> np.ndarray:
    _numeric_matrix(X, "X"); _numeric_matrix(Z, "Z")
    if X.shape[1] != Z.shape[1]:
        raise ValueError("X和Z特征数必须一致")
    power = _distance_power(p); feature_weights = _feature_weights(weights, X.shape[1], "weights")
    powered = np.abs(X.astype(float)[:, None, :] - Z.astype(float)[None, :, :]) ** power
    return np.sum(powered * feature_weights[None, None, :], axis=2) ** (1.0 / power)


def fit_vdm(categorical_data: np.ndarray, group_labels: np.ndarray, *,
            alpha: float = 0.0) -> dict[str, object]:
    """按每个无序属性值估计P(group|value)，供VDM查询使用。"""
    _categorical_matrix(categorical_data, "categorical_data")
    _labels(group_labels, len(categorical_data), "group_labels")
    if (isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real)
            or not np.isfinite(alpha) or alpha < 0):
        raise ValueError("alpha必须是有限非负数")
    classes, class_inverse = np.unique(group_labels, return_inverse=True)
    feature_values = []; probabilities = []
    for feature in range(categorical_data.shape[1]):
        values, value_inverse = np.unique(categorical_data[:, feature], return_inverse=True)
        counts = np.zeros((len(values), len(classes)), dtype=float)
        np.add.at(counts, (value_inverse, class_inverse), 1.0)
        denominator = counts.sum(axis=1, keepdims=True) + float(alpha) * len(classes)
        probabilities.append((counts + float(alpha)) / denominator)
        feature_values.append(values.copy())
    return {"classes": classes.copy(), "feature_values": tuple(feature_values),
            "probabilities": tuple(probabilities), "alpha": float(alpha)}


def _validate_vdm_model(model: dict[str, object]) -> None:
    if not isinstance(model, dict) or set(model) != {"classes", "feature_values", "probabilities", "alpha"}:
        raise ValueError("VDM model键集合无效")


def _vdm_indices(model: dict[str, object], X: np.ndarray) -> np.ndarray:
    _validate_vdm_model(model); _categorical_matrix(X, "X")
    if X.shape[1] != len(model["feature_values"]):
        raise ValueError("无序属性列数与VDM模型不一致")
    result = np.empty(X.shape, dtype=int)
    for feature, values in enumerate(model["feature_values"]):
        mapping = {value: index for index, value in enumerate(values.tolist())}
        for row, value in enumerate(X[:, feature].tolist()):
            if value not in mapping:
                raise ValueError("查询数据含训练时未见的无序属性值")
            result[row, feature] = mapping[value]
    return result


def pairwise_vdm(X: np.ndarray, Z: np.ndarray, model: dict[str, object], *,
                 p: float = 1.0, weights: np.ndarray | None = None) -> np.ndarray:
    """聚合各无序属性上条件分布差异，返回带p次根的成对VDM。"""
    power = _distance_power(p)
    X_indices = _vdm_indices(model, X); Z_indices = _vdm_indices(model, Z)
    if X.shape[1] != Z.shape[1]:
        raise ValueError("X和Z无序属性数必须一致")
    feature_weights = _feature_weights(weights, X.shape[1], "weights")
    total = np.zeros((len(X), len(Z)), dtype=float)
    for feature, probabilities in enumerate(model["probabilities"]):
        differences = np.abs(probabilities[X_indices[:, feature], None, :]
                             - probabilities[None, Z_indices[:, feature], :]) ** power
        total += feature_weights[feature] * np.sum(differences, axis=2)
    return total ** (1.0 / power)


def pairwise_mixed_distance(
    X_numeric: np.ndarray, Z_numeric: np.ndarray,
    X_categorical: np.ndarray, Z_categorical: np.ndarray,
    vdm_model: dict[str, object], *, p: float = 2.0,
    numeric_weights: np.ndarray | None = None,
    categorical_weights: np.ndarray | None = None,
) -> np.ndarray:
    """把有序数值属性的差与无序属性VDM放进同一个加权p次和。"""
    _numeric_matrix(X_numeric, "X_numeric"); _numeric_matrix(Z_numeric, "Z_numeric")
    _categorical_matrix(X_categorical, "X_categorical"); _categorical_matrix(Z_categorical, "Z_categorical")
    if len(X_numeric) != len(X_categorical) or len(Z_numeric) != len(Z_categorical):
        raise ValueError("数值与无序属性的样本行数必须对应")
    power = _distance_power(p)
    numeric = pairwise_weighted_minkowski(X_numeric, Z_numeric, p=power, weights=numeric_weights) ** power
    categorical = pairwise_vdm(X_categorical, Z_categorical, vdm_model, p=power,
                               weights=categorical_weights) ** power
    return (numeric + categorical) ** (1.0 / power)
