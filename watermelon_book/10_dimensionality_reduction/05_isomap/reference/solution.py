"""参考实现：k近邻图、Floyd最短路和经典MDS组成的Isomap。"""
import numpy as np

def _matrix(X,name,finite=True):
    if not isinstance(X,np.ndarray) or X.ndim!=2 or 0 in X.shape or not np.issubdtype(X.dtype,np.number) or (finite and not np.all(np.isfinite(X))): raise ValueError(f"{name}必须是非空数值二维数组")

def pairwise_euclidean(X):
    _matrix(X,"X"); diff=X.astype(float)[:,None,:]-X.astype(float)[None,:,:]
    return np.sqrt(np.sum(diff*diff,axis=2))

def knn_graph(X,n_neighbors):
    _matrix(X,"X"); n=X.shape[0]
    if isinstance(n_neighbors,(bool,np.bool_)) or not isinstance(n_neighbors,(int,np.integer)) or n_neighbors<=0 or n_neighbors>=n: raise ValueError("n_neighbors必须是1到n-1之间的整数")
    distances=pairwise_euclidean(X); directed=np.full((n,n),np.inf); np.fill_diagonal(directed,0)
    for i in range(n):
        candidates=np.arange(n); candidates=candidates[candidates!=i]
        order=np.argsort(distances[i,candidates],kind="stable")[:int(n_neighbors)]
        chosen=candidates[order]; directed[i,chosen]=distances[i,chosen]
    graph=np.minimum(directed,directed.T); np.fill_diagonal(graph,0)
    return graph

def floyd_warshall(graph):
    _matrix(graph,"graph",finite=False)
    if graph.shape[0]!=graph.shape[1] or np.any(np.isnan(graph)) or np.any(graph<0) or not np.allclose(graph,graph.T,equal_nan=True) or not np.allclose(np.diag(graph),0): raise ValueError("graph必须是非负对称、对角为0的方阵")
    shortest=graph.astype(float).copy(); n=graph.shape[0]
    for k in range(n): shortest=np.minimum(shortest,shortest[:,k,None]+shortest[None,k,:])
    return shortest

def _mds(distances,n_components):
    n=distances.shape[0]; J=np.eye(n)-np.ones((n,n))/n; B=-.5*J@(distances**2)@J; B=(B+B.T)/2
    values,vectors=np.linalg.eigh(B); order=np.argsort(values)[::-1]; values=values[order]; vectors=vectors[:,order]
    selected=np.maximum(values[:n_components],0); coordinates=vectors[:,:n_components]*np.sqrt(selected)[None,:]
    for j in range(n_components):
        pivot=int(np.argmax(np.abs(coordinates[:,j])))
        if coordinates[pivot,j]<0: coordinates[:,j]*=-1
    return coordinates,values

def fit_isomap(X,n_neighbors,n_components):
    _matrix(X,"X"); n=X.shape[0]
    if isinstance(n_components,(bool,np.bool_)) or not isinstance(n_components,(int,np.integer)) or n_components<=0 or n_components>n: raise ValueError("n_components必须是1到样本数之间的整数")
    euclidean=pairwise_euclidean(X); graph=knn_graph(X,n_neighbors); geodesic=floyd_warshall(graph)
    if not np.all(np.isfinite(geodesic)): raise ValueError("近邻图不连通，请增大n_neighbors")
    coordinates,eigenvalues=_mds(geodesic,int(n_components))
    return {"coordinates":coordinates,"euclidean_distances":euclidean,"graph":graph,"geodesic_distances":geodesic,"eigenvalues":eigenvalues}
