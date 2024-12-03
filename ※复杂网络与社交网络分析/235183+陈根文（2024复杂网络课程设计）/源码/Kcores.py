import networkx as nx
import numpy as np
import scipy as sp
from operator import itemgetter

def K_cores(G, G2):
    # 通过各个节点的核数发现重要节点
    core = 1
    flag = 1
    node_cores = {}

    for node in G.nodes():
        node_cores[node] = -1

    while(G.number_of_nodes() != 0):
        # 循环去除
        while(flag == 1):
            # 还有变化
            flag = 0
            nodes = list(G.nodes())
            for node in nodes:
                if G.degree(node) <= core:
                    node_cores[node] = core
                    flag = 1
                    G.remove_node(node)
        flag = 1
        core += 1

    # print(node_cores)
    cores1 = sorted(node_cores.items(), key=itemgetter(1), reverse=True)  # 排序

    # ncores = cores1[:100]
    # f = open("K-coresRank.txt", 'w')
    # f.write("重要节点排序:\n" + str(ncores)+"\n\n各节点的K-coresRank值:\n" + str(cores1))
    # f.close()

    return cores1

