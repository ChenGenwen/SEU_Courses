import networkx as nx
import numpy as np
import scipy as sp


def calculate_pagerank(graph):
    # 计算 PageRank 值
    #pagerank_scores = nx.pagerank(graph)
    pagerank_scores = func_pagerank(graph)
    return pagerank_scores

# 获取 PageRank 值最高的 k 个节点
def top_influential_nodes(scores):
    top_nodes = sorted(scores, key=scores.get, reverse=True)
    return top_nodes

def func_pagerank(G):
    alpha=0.85
    personalization=None
    max_iter=100
    tol=1.0e-6
    nstart=None
    weight="weight"
    dangling=None

    N = len(G)
    if N == 0:
        return {}
    nodelist = list(G)
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight=weight, dtype=float)
    S = A.sum(axis=1)
    S[S != 0] = 1.0 / S[S != 0]

    Q = sp.sparse.csr_array(sp.sparse.spdiags(S.T, 0, *A.shape))
    A = Q @ A
    # 初始向量
    if nstart is None:
        x = np.repeat(1.0 / N, N)
    else:
        x = np.array([nstart.get(n, 0) for n in nodelist], dtype=float)
        x /= x.sum()

    # 个性化向量
    if personalization is None:
        p = np.repeat(1.0 / N, N)
    else:
        p = np.array([personalization.get(n, 0) for n in nodelist], dtype=float)
        if p.sum() == 0:
            raise ZeroDivisionError
        p /= p.sum()
     # 悬空节点
    if dangling is None:
        dangling_weights = p
    else:
         # 将悬空节点的字典转换为按nodelist顺序的数组
        dangling_weights = np.array([dangling.get(n, 0) for n in nodelist], dtype=float)
        dangling_weights /= dangling_weights.sum()
    is_dangling = np.where(S == 0)[0]

    # 幂迭代：进行最多max_iter次迭代
    for _ in range(max_iter):
        xlast = x
        x = alpha * (x @ A + sum(x[is_dangling]) * dangling_weights) + (1 - alpha) * p
         # 检查收敛性，l1范数
        err = np.absolute(x - xlast).sum()
        if err < N * tol:
            return dict(zip(nodelist, map(float, x)))
    raise nx.PowerIterationFailedConvergence(max_iter)

#打印前100个得分最高的节点
def my_PageRank(graph):
     # 计算 PageRank 值
    pagerank_scores = calculate_pagerank(graph)
    # 获取 PageRank 值最高的 k 个节点
    top_nodes = top_influential_nodes(pagerank_scores)

    # # 设置要获取的影响力最大节点数
    # top_nodes_k = top_nodes[:100]
    # # 打印结果Top {k} Influential Nodes
    # f = open("PageRank.txt", 'w')
    # for node in top_nodes_k:
    #     f.write(f"Node:{node}  ,  PageRank Score:{pagerank_scores[node]}\n")

    return top_nodes

   

   