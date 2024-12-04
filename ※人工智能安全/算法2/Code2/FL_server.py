import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import copy
import time
import argparse
import random
import math

import numpy as np
from FL_clients import ClientsManager
from torch.utils.tensorboard import SummaryWriter 
from torchvision.transforms.transforms import Grayscale
from utils import get_outputs_filename, get_dataset, get_network, ParamDiffAug, evaluate_synset, evaluate_synset_mapping

parser = argparse.ArgumentParser(description='Parameter Processing')
parser.add_argument('--Dataset', default='SVHN', type=str, help='CIFAR10, FashionMNIST, SVHN, CIFAR100')
parser.add_argument('--model', type=str, default='ConvNet', help='model')
parser.add_argument('--ur', default=0.1, type=float, help='')
parser.add_argument('--lr', default=1e-2, type=float, help='')
parser.add_argument('--device', default=0, type=int, help='')
parser.add_argument('--Clients_Num', default=10, type=int, help='')
parser.add_argument('--Local_Batchsize', default=256, type=int, help='')
parser.add_argument('--FL_Rounds', default=20, type=int, help='')
parser.add_argument('--noiid', action='store_true', help='')
parser.add_argument('--method', type=str, default='myFedDM', help='myFedDM, FedDM, FL, FedProx, FedNova, ILT')
parser.add_argument('--mapping', action='store_true', help='applying mapping model')
parser.add_argument('--ipc', type=int, default=10, help='image(s) per class')
parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')
parser.add_argument('--num_eval', type=int, default=1, help='the number of evaluating randomly initialized models')
parser.add_argument('--epoch_mapping_model_train', type=int, default=10, help='epochs to train a model with synthetic data')
parser.add_argument('--epoch_eval_train', type=int, default=500, help='epochs to train a model with synthetic data')
parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')
parser.add_argument('--lr_img', type=float, default=1, help='learning rate for updating synthetic images')
parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
parser.add_argument('--init', type=str, default='real', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
parser.add_argument('--alpha', type=float, default=0.1, help='distance metric')
parser.add_argument('--conv_num', type=int, default=2, help='distance metric')
parser.add_argument('--kernel_num', type=int, default=16, help='distance metric')
args = parser.parse_args()

args.dsa_param = ParamDiffAug()
args.dsa = True

#检测gpu
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:'+str(args.device))
print("Running on %s" % device)

#是否使用隐射模块
args.mapping = True if args.method == 'myFedDM' else False
noiid_str = '-NoIID{}'.format(args.alpha) if args.noiid else '-IID'

#定义实验文件夹名
args.save_path = get_outputs_filename(f'test-{args.method}{noiid_str}-{args.Dataset}-ipc{args.ipc}-{args.conv_num}conv-{args.kernel_num}kernel-{args.init}')
writer = SummaryWriter(args.save_path+'/logs')



channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.Dataset)
args.channel, args.im_size, args.num_classes = channel, im_size, num_classes

#根据kernel_num计算输入的im_size
w = int(math.sqrt(args.kernel_num)*8)
args.im_size = (w, w)
global_net = get_network(args.model, 1 if args.mapping else channel, num_classes, args.im_size).to(args.device)


CM = ClientsManager(args)

print(CM.clients)

# 客户端发送DC数据，服务端不发送最新模型
# #开始进行联邦学习
# image_syn_eval_all = torch.empty(0, 1 if args.mapping else args.channel, args.im_size[0], args.im_size[1]).to(args.device)
# label_syn_eval_all = torch.empty(0).to(args.device)

# for i in range(args.FL_Rounds):
#     print(' ')
#     print('# 开始第 %d 轮联邦学习 #' % (i+1))

#     for j in range(args.Clients_Num):
#     #客户端执行本地DC
#         CM.clients[j].localDC(args)

#     #拼接所有客户端的蒸馏数据集
#     print('\n##########   开始全局模型测试   ##########\n')
#     image_syn_eval = [copy.deepcopy(client.image_syn.detach()) for client in CM.clients]
#     label_syn_eval = [copy.deepcopy(client.label_syn.detach()) for client in CM.clients]
#     image_syn_eval = torch.cat(image_syn_eval)
#     label_syn_eval = torch.cat(label_syn_eval)
#     image_syn_eval_all = torch.cat([image_syn_eval_all, image_syn_eval])
#     label_syn_eval_all = torch.cat([label_syn_eval_all, label_syn_eval])
#     print(f'生成数据集的数据量为 {len(label_syn_eval_all)}')

#     testloaders = [client.test_loader for client in CM.clients]
#     if args.mapping:
#         mapping_models = [copy.deepcopy(client.mapping_model.conv) for client in CM.clients]
#         _, acc_train, acc_test = evaluate_synset_mapping(1, global_net, image_syn_eval_all, label_syn_eval_all, testloaders, mapping_models, args)
#     else:
#         # _, acc_train, acc_test = evaluate_synset(1, global_net, image_syn_eval_all, label_syn_eval_all, testloaders, args)
#         _, acc_train, acc_test = evaluate_synset_mapping(1, global_net, image_syn_eval_all, label_syn_eval_all, testloaders, None, args)


#客户端发送DC数据，服务器发送最新模型
#开始进行联邦学习
if args.method == 'myFedDM':
    for i in range(args.FL_Rounds):
        print(' ')
        print('# 开始第 %d 轮联邦学习 #' % (i+1))
        accs_local = []

        for j in range(args.Clients_Num):
        #客户端执行本地DC
            acc_local = CM.clients[j].localDC2(copy.deepcopy(global_net.state_dict()), args)
            accs_local.append(acc_local)
        accs_avg_localtrain = np.mean(accs_local)

        #拼接所有客户端的蒸馏数据集
        print('\n##########   开始全局模型测试   ##########\n')
        image_syn_eval = [copy.deepcopy(client.image_syn.detach()) for client in CM.clients]
        label_syn_eval = [copy.deepcopy(client.label_syn.detach()) for client in CM.clients]
        image_syn_eval = torch.cat(image_syn_eval)
        label_syn_eval = torch.cat(label_syn_eval)
        print(f'生成数据集的数据量为 {len(label_syn_eval)}')

        testloaders = [client.test_loader for client in CM.clients]
        if args.mapping:
            mapping_models = [copy.deepcopy(client.mapping_model.conv) for client in CM.clients]
            _, acc_train, acc_test, acc_avg_clients = evaluate_synset_mapping(1, global_net, image_syn_eval, label_syn_eval, testloaders, mapping_models, args)
        else:
            # _, acc_train, acc_test = evaluate_synset(1, global_net, image_syn_eval_all, label_syn_eval_all, testloaders, args)
            _, acc_train, acc_test = evaluate_synset_mapping(1, global_net, image_syn_eval, label_syn_eval, testloaders, None, args)

        print('local avg acc after local training  %.4f' % accs_avg_localtrain)
        writer.add_scalar('acc_test', acc_test, i+1)
        writer.add_scalar('acc_avg_clients', acc_avg_clients, i+1)
        writer.add_scalar('accs_avg_localtrain', accs_avg_localtrain, i+1)

if args.method == 'FedDM':
    for i in range(args.FL_Rounds):
        print(' ')
        print('# 开始第 %d 轮联邦学习 #' % (i+1))

        for j in range(args.Clients_Num):
        #客户端执行本地DC
            CM.clients[j].FedDM(copy.deepcopy(global_net.state_dict()), args)

        #拼接所有客户端的蒸馏数据集
        print('\n##########   开始全局模型测试   ##########\n')
        image_syn_eval = [copy.deepcopy(client.image_syn.detach()) for client in CM.clients]
        label_syn_eval = [copy.deepcopy(client.label_syn.detach()) for client in CM.clients]
        image_syn_eval = torch.cat(image_syn_eval)
        label_syn_eval = torch.cat(label_syn_eval)
        print(f'生成数据集的数据量为 {len(label_syn_eval)}')

        testloaders = [client.test_loader for client in CM.clients]

        _, acc_train, acc_test, acc_avg_clients = evaluate_synset_mapping(1, global_net, image_syn_eval, label_syn_eval, testloaders, None, args)

        writer.add_scalar('acc_test', acc_test, i+1)
        writer.add_scalar('acc_avg_clients', acc_avg_clients, i+1)

elif args.method == 'FL' or args.method == 'FedProx' or args.method == 'FedNova':

    #计算梯度聚合比例
    grad_ratio = [1/args.Clients_Num]*args.Clients_Num
    if args.method == 'FedNova':
        client_datanum = [len(c.local_dataset) for c in CM.clients]
        total_num = np.sum(client_datanum)
        grad_ratio = client_datanum/total_num

    for i in range(args.FL_Rounds):
        print(' ')
        print('# 开始第 %d 轮联邦学习 #' % (i+1))

        aggregated_grads = None

        for j in range(args.Clients_Num):
        #客户端执行本地DC
            local_grads = CM.clients[j].fl(global_net.state_dict(), args)

            #统计梯度
            if aggregated_grads ==None:
                aggregated_grads = {}
                for k, v in local_grads.items():
                    aggregated_grads[k] = v*grad_ratio[j]
            else:
                for k, v in local_grads.items():
                    aggregated_grads[k] += v*grad_ratio[j]

        #FedNova根据本地sgd_iters放缩梯度
        if args.method == 'FedNova':
            tau_eff = np.sum([grad_ratio[l]*CM.clients[l].sgd_iters for l in range(args.Clients_Num)])
            for k,v in aggregated_grads.items():
                aggregated_grads[k] = v*tau_eff

        #更新全局模型
        global_params = global_net.state_dict()
        for k, v in global_params.items():
            global_params[k] += aggregated_grads['_module.' + k]
        global_net.load_state_dict(global_params)


        testloaders = [client.test_loader for client in CM.clients]

        #开始测试模型
        loss_avg, acc_avg, num_exp = 0, 0, 0
        global_net.eval()
        criterion = nn.CrossEntropyLoss().to(args.device)

        accs = []
        for j in range(len(testloaders)):
            acc_avg_client, num_exp_client = 0, 0
            for i_batch, datum in enumerate(testloaders[j]):
                img = datum[0].float().to(args.device)

                lab = datum[1].long().to(args.device)
                n_b = lab.shape[0]

                output = global_net(img)
                loss = criterion(output, lab)
                acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

                loss_avg += loss.item()*n_b
                acc_avg += acc
                num_exp += n_b
                acc_avg_client += acc
                num_exp_client += n_b
            acc_client = acc_avg_client/num_exp_client
            accs.append(acc_client)

        loss_avg /= num_exp
        acc_avg /= num_exp
        acc_avg_clients = np.mean(accs)

        print('global test acc = %.4f' % (acc_avg))
        
        writer.add_scalar('acc_test', acc_avg, i+1)
        writer.add_scalar('acc_avg_clients', acc_avg_clients, i+1)

elif args.method == 'ILT':
    for i in range(args.FL_Rounds):
        print(' ')
        print('# 开始第 %d 轮本地单独训练 #' % (i+1))

        accs_local = []
        for j in range(args.Clients_Num):

            acc_local = CM.clients[j].individual_local_train(args)
            accs_local.append(acc_local)
        accs_avg_localtrain = np.mean(accs_local)

        print('local avg acc after local training  %.4f' % accs_avg_localtrain)
        writer.add_scalar('accs_avg_localtrain', accs_avg_localtrain, i+1)

time.sleep(3)
    

