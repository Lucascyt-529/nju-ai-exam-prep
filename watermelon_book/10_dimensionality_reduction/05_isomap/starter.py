"""学生练习：Isomap近邻图、最短路与嵌入。"""
import numpy as np

def pairwise_euclidean(X: np.ndarray) -> np.ndarray: raise NotImplementedError
def knn_graph(X: np.ndarray, n_neighbors: int) -> np.ndarray: raise NotImplementedError
def floyd_warshall(graph: np.ndarray) -> np.ndarray: raise NotImplementedError
def fit_isomap(X: np.ndarray, n_neighbors: int, n_components: int) -> dict[str, np.ndarray]: raise NotImplementedError
