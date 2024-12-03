import networkx as nx
import os
import random
import data_process,PageRank,HITS,Kcores,Degree,Conv
from LT import *
from operator import itemgetter
from matplotlib import pyplot as plt



def cal_influence(G, seeds, k=1):
    # 通过LT线性阈值模型，计算单位时间内top-k节点的影响力范围
    layers = linear_threshold(G, seeds, k)      # seeds为种子节点集合
    length = 0
    for i in range(1, len(layers)):
        length += len(layers[i])        # 计算每个种子节点激活节点的个数之和
    return length

def sort(dic):
    dic1 = sorted(dic.items(), key=itemgetter(1), reverse=True)  # 排序
    dnode = [e[0] for e in dic1]   # 取节点
    return dnode

'''预分析'''
def Primal_Info(myGraph: nx.Graph,fn):
    # 输出图的基本结构
    f = open("attribute_" + fn + ".txt", 'w')
    f.write("节点数: " + str(myGraph.number_of_nodes()))
    f.write("\n\n连边数: " + str(myGraph.number_of_edges()))
    f.write("\n\n网络密度: " + str(nx.density(myGraph)))
    f.write("\n\n平均网络的聚类系数: " + str(nx.average_clustering(nx.DiGraph(myGraph))))
    f.write("\n\n整个网络的聚类系数: " + str(nx.transitivity(myGraph)))
    f.close()

if  __name__ == "__main__":

    # if not os.path.exists('./data/social.txt'):
    #     # 第一次运行时读取数据并创建图
    #     edgelist_file =  "data/higgs-social_network.edgelist"
    #     myGraph = data_process.read_graph_from_edgelist(edgelist_file)
    #     #绘制度分布散点图
    #     data_process.draw_degree_plot(myGraph)
    #     # 保存图到文件
    #     data_process.save_graph_to_file(myGraph, "./data/social.txt")
    #     #清洗节点并得到txt文件
    #     Files = ['retweet','reply','mention']
    #     for data_file_name in Files:
    #         if not (os.path.exists('./data/retweet.txt') and os.path.exists('./data/reply.txt')
    #                 and os.path.exists('./data/mention.txt')):
    #             data_process.clean_node(data_file_name,myGraph)

    # else:
    #     #重新加载图
    #     myGraph = data_process.load_graph_from_file("./data/social.txt")
    myGraph = nx.read_edgelist("./data/higgs-social_network.edgelist",create_using=nx.DiGraph())

    #预分析
    # Primal_Info(myGraph,"social")
    # G1 = data_process.load_graph_from_file("./data/mention.txt")  
    # Primal_Info(G1,"mention")
    # G2 = data_process.load_graph_from_file("./data/reply.txt")
    # Primal_Info(myGraph,"reply")
    # G3 = data_process.load_graph_from_file("./data/retweet.txt")
    # Primal_Info(myGraph,"retweet")


    #compare_all_methods比较所有方法
    atr=dict(myGraph.out_degree())

    conv1=Conv.conv(myGraph,atr)
    convself=dict()
    convself2=dict()
    convself3=dict()
    for k in atr.keys():
        if k in conv1.keys():
            convself2[k] = atr[k]+0.5*conv1[k]
            convself[k] = atr[k] + 0.2*conv1[k]
            convself3[k] = atr[k]+conv1[k]
        else:
            convself2[k] = atr[k]
            convself[k] = atr[k]
            convself3[k] = atr[k]
    # 输出图的基本结构
    k = 10
    x = []
    y1 = []
    y2 = []
    y3 = []
    y4=[]
    y5=[]
    y6=[]
    y7=[]
    y8=[]

    PageRank_nodes=PageRank.my_PageRank(myGraph)
    HITS_hubs,HITS_authorities = HITS.get_HITS_Nodes(myGraph)
    myGraph_copy = myGraph.copy()
    kcores = Kcores.K_cores(myGraph_copy, myGraph)  
    degree_centality = Degree.degree_centrality(myGraph)

    nc1=sort(convself)
    nc2=sort(convself2)
    nc3=sort(convself3)


    for seeds_num in range(200, 410,50):
        seed_node = str(seeds_num)
        if seed_node not in myGraph:
            print(f"Seed node {seed_node} is not in the graph.")
            continue 
        print('='*20+seed_node+"\n")
        x.append(seed_node)

        print("degree_centality[1]={}\n".format(degree_centality[1]))
        degree_centality_seeds = [e[0] for e in degree_centality[:seeds_num]] 

        print("kcores[1]={}\n".format(kcores[1]))
        kcores_seeds = [k[0] for k in kcores[:seeds_num]] 

        print("PageRank_nodes[1]={}\n".format(PageRank_nodes[1]))
        PageRank_nodes_seeds = [p[0] for p in PageRank_nodes[:seeds_num]] 

        print("HITS_hubs[1]={}\n".format(HITS_hubs[1]))
        HITS_hubs_seeds = [h[0] for h in HITS_hubs[:seeds_num]] 

        print("HITS_authorities[1]={}\n".format(HITS_authorities[1]))
        HITS_authorities_seeds = [a[0] for a in HITS_authorities[:seeds_num]] 

        print("nc1[1]={}\n".format(nc1[1]))
        nc1_seeds = [n[0] for n in nc1[:seeds_num]] 
        nc2_seeds = [n[0] for n in nc2[:seeds_num]] 
        nc3_seeds = [n[0] for n in nc3[:seeds_num]] 

        y1.append(cal_influence(myGraph, degree_centality_seeds, k))
        y2.append(cal_influence(myGraph, kcores_seeds, k))
        y3.append(cal_influence(myGraph, nc1_seeds, k))
        y4.append(cal_influence(myGraph, nc2_seeds, k))
        y5.append(cal_influence(myGraph, PageRank_nodes_seeds, k))
        y6.append(cal_influence(myGraph, HITS_hubs_seeds, k))
        y7.append(cal_influence(myGraph, HITS_authorities_seeds, k))
        y8.append(cal_influence(myGraph, nc3_seeds, k))
    l1 = plt.plot(x, y1, 'r--', label='Degree')
    l2 = plt.plot(x, y2, 'c--', label='Kcores')
    l3 = plt.plot(x, y3, 'b--', label='Conv0.2')
    l4 = plt.plot(x, y4, 'k--', label='Conv0.5')
    l8 = plt.plot(x, y8, 'y--', label='Conv1')
    l5 = plt.plot(x, y5, 'g--', label='Pagerank')
    l6 = plt.plot(x, y6, 'm--', label='HITS-H')
    l7 = plt.plot(x, y7, 'm--', label='HITS-A')
    print("end")
    plt.plot(x, y1, 'ro-', x, y2, 'c+-', x, y3, 'b^-', x, y4, 'ks-' , x, y5, 'gp-', x, y6, 'm1-', x, y7, 'm2-', x, y8, 'y3-')
    plt.legend()
    plt.savefig('Compare.png')
    plt.show()
