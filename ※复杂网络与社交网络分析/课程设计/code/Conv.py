import networkx as nx

def conv(G,atr):#atr是计算出来的degree
    conv={}#建立字典
    for edge in G.edges():
        if edge[0] in conv.keys():
            if edge[1] in atr.keys():
                conv[edge[0]] += atr[edge[1]] / atr[edge[0]]
        else:
            if edge[1] in atr.keys():
                conv[edge[0]] = atr[edge[1]] / atr[edge[0]]
            else:
                conv[edge[0]]=0
    return conv