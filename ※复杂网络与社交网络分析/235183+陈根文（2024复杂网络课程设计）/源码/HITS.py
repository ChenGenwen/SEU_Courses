import networkx as nx
import numpy as np
import scipy as sp
from operator import itemgetter

def calculate_hits(graph):
    # 计算 HITS 算法得到 Hub 和 Authority 值
    #hits_scores = nx.hits(graph)
    hits_scores = func_HITS(graph)
    return hits_scores

def func_HITS(G):
    max_iter=100
    tol=1.0e-8
    nstart=None
    normalized=True
    if len(G) == 0:
        return {}, {}
    A = nx.adjacency_matrix(G, nodelist=list(G), dtype=float)

    if nstart is None:
        _, _, vt = sp.sparse.linalg.svds(A, k=1, maxiter=max_iter, tol=tol)
    else:
        nstart = np.array(list(nstart.values()))
        _, _, vt = sp.sparse.linalg.svds(A, k=1, v0=nstart, maxiter=max_iter, tol=tol)

    a = vt.flatten().real
    h = A @ a
    if normalized:
        h /= h.sum()
        a /= a.sum()
    hubs = dict(zip(G, map(float, h)))
    authorities = dict(zip(G, map(float, a)))
    return hubs, authorities

def top_influential_nodes(scores):
    # 获取 Hub 或 Authority 值最高的 k 个节点
    top_nodes = sorted(scores, key=scores.get, reverse=True)
    return top_nodes

def my_HITS(graph):
    # 计算 HITS 算法得到 Hub 和 Authority 值
    hits_scores = calculate_hits(graph)
    # 获取 Authority 值最高的 k 个节点
    authority_nodes = top_influential_nodes(hits_scores[1])

    # 打印结果Top {k} Influential Nodes based on Authority:
    # top_authority_nodes = authority_nodes[:100]
    # f = open("HITS.txt", 'w')
    # for node in top_authority_nodes:
    #     f.write(f"Node:{node}  ,  Authority Score:{hits_scores[1][node]}\n")

    return authority_nodes

def get_HITS_Nodes(G):
    h,a=func_HITS(G)
    h1 = sorted(h.items(), key=itemgetter(1), reverse=True)  # 排序
    a1 = sorted(a.items(), key=itemgetter(1), reverse=True)  # 排序
    hhr = [e[0] for e in h1]   # 取节点
    har = [e[0] for e in a1]   # 取节点
    return hhr,har

