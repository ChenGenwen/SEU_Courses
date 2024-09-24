import networkx as nx
import numpy as np
import scipy as sp
from operator import itemgetter

def degree_centrality(G):
    # 度中心性
    dc = nx.degree_centrality(G)  # degree
    dc1 = sorted(dc.items(), key=itemgetter(1), reverse=True)  # 排序

    # ndc = [e[0] for e in dc1[:100]]   # 取节点
    # f = open("DegreeRank.txt", 'w')
    # f.write("重要节点排序:\n" + str(ndc) + "\n\n各节点的度数中心性:\n" + str(dc1))
    # f.close()

    return dc1