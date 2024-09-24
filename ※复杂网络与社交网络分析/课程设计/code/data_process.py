import networkx as nx
import matplotlib.pyplot as plt
from operator import itemgetter
import numpy as np
from scipy import sparse
import math
import os
import random

# 读取 edgelist 文件并创建图
def read_graph_from_edgelist(file_path):
    # 读文件，制图
    f = open(file_path, 'r')
    # 建立有向图
    G = nx.DiGraph()
    
    lines = f.readlines()
    # 一次性读入整个文件，加快处理速度
    for line in lines:
        line = line.strip()     # 去掉结尾换行符
        line = line.split()     # 以空格或回车分割句子
        G.add_edge(int(line[0]), int(line[1]))
        # 建立边关系
    f.close()
    return G

# 保存图到文件
def save_graph_to_file(graph, file_path):
    nx.write_edgelist(graph, file_path)

# 重新加载已经存入txt的图
def load_graph_from_file(file_path):
    G = nx.read_edgelist(file_path,create_using=nx.DiGraph())
    return G

#绘制度分布图
def draw_degree_plot(myGraph):
    # 获取图的度分布直方图
    Degree_Hist = nx.degree_histogram(myGraph)
    # 初始化一个列表，用于存储处理后的度分布数据
    Degree_Hist = [Degree_Hist, [], []]
    # 遍历度分布直方图，处理为 (度数, 频数) 的形式
    for i in range(0, len(Degree_Hist[0])):
        if Degree_Hist[0][i] == 0:
            continue
        Degree_Hist[1].append(i)    # 存储度数
        Degree_Hist[2].append(Degree_Hist[0][i])    # 存储对应的频数
     # 设置中文显示和字体
    plt.rcParams["font.sans-serif"] = "SimHei"
    plt.rcParams["axes.unicode_minus"] = False
    # 创建一个图形，设置图形大小
    fig = plt.figure(figsize=(12, 6))
     # 在图形中创建两个子图，分别绘制原始度数和对数化后的度数
    axes1 = plt.subplot(121)
    axes1.scatter([i for i in Degree_Hist[1]], [math.log(i + 1) for i in Degree_Hist[2]],
                  s=0.5, ) # 绘制原始度数的散点图
    axes1.set_xlabel("Degree")
    axes1.set_ylabel("log(count+1)")
    axes2 = plt.subplot(122)
    axes2.scatter([math.log(i) for i in Degree_Hist[1]], [math.log(i + 1) for i in Degree_Hist[2]],
                  s=0.8, )  # 绘制对数化后的度数的散点
    axes2.set_xlabel("log(Degree)")
    axes2.set_ylabel("log(count+1)")
    plt.suptitle("度分布散点图")
    plt.savefig("./pic/度分布散点图.png") # 保存图形到文件

#清洗节点
def clean_node(data_file_name, G: nx.DiGraph):
    print('=' * 15 + f"数据清洗" + '=' * 15)
    f1 = './data/{}.txt'.format(data_file_name)
    f_clean = open(f1, 'w', encoding='utf-8')
    f2 = './data/higgs-{}_network.edgelist'.format(data_file_name)
    with open(f2, 'r', encoding='utf-8') as f:
        for line in f:
            l = line.split(' ')
            n1, n2 = int(l[0]), int(l[1])
            if n1 in G.nodes and n2 in G.nodes:
                f_clean.write(line)

#统计边列表,num表示图的节点数目,防止过大不方便绘制
def stat_node(data_file_name, num):
    f = './data/{}.txt'.format(data_file_name)
    edgeList = []
    count = 0
    with open(f, 'r', encoding='utf-8') as f:
        for line in f:
            if count >= num:
                return edgeList
            edge = [int(one) for one in line.strip('\n').split(' ')]
            edgeList.append(tuple(edge))
            count+=1
    return edgeList

#创建图
def create_graph(data_file_name, num):
    graph = nx.DiGraph()
    edgeList = stat_node(data_file_name, num)
    graph.add_weighted_edges_from(edgeList)
    return graph



#画出转发关系图
# def draw_pic(data_file_name, num):
#     # 获得图
#     graph = create_graph(data_file_name, num)

#     fig, ax = plt.subplots(figsize=(15, 9))
#     ax.axis('off')
#     plt.title(data_file_name+" Networks", fontsize=20)

#     # 画图1
#     plot_options = {"node_size": 6, "with_labels": False, "width": 0.15}
#     # plot_options = {"node_size": 6, "with_labels": False,
#     #                 "width": [float(d['weight'] / 20) for (u, v, d) in graph.edges(data=True)]}
#     nx.draw_networkx(graph, pos=nx.random_layout(graph), ax=ax, **plot_options)

#     # 画图2
#     # pos = nx.random_layout(graph)
#     # nx.draw(graph, pos, with_labels=True, alpha=0.5)
#     # labels = nx.get_edge_attributes(graph, 'weight')
#     # nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
#     # nx.draw_networkx_edges(graph, pos, width=[float(d['weight'] / 1.5) for (u, v, d) in graph.edges(data=True)])

#     plt.savefig('./pic/pic-{}.png'.format(data_file_name))
#     # plt.show()




  


    

