from torch.utils.data import DataLoader, random_split
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import time
from utils import get_outputs_filename, get_dataset, get_network, get_time, DiffAugment
from opacus import PrivacyEngine

class Client():
    def __init__(self, id, local_dataset, args) -> None:
        self.device = args.device
        self.args = args
        self.id = id
        self.DCnum = 0
        self.distilled_dataset = None
        self.local_dataset = local_dataset
        self.image_syn = None
        self.label_syn = None
        self.images_all = None
        self.labels_all = None
        self.indices_class = None
        self.syn_img_num = None
        self.syn_img_index = None
        self.local_net = None
        self.sgd_iters = 0

        #将local_dataset按8：2划分为训练和测试集
        trainset_size = int(len(self.local_dataset)*0.8)
        splits = random_split(self.local_dataset, [trainset_size, len(self.local_dataset)-trainset_size])
        self.dst_train = splits[0]
        self.train_loader = DataLoader(splits[0], batch_size=args.Local_Batchsize, shuffle=True)
        self.test_loader = DataLoader(splits[1], batch_size=args.Local_Batchsize, shuffle=False)

        #本地隐射模块
        self.mapping_model = get_network(model='CIFAR10CNN', channel=args.channel, num_classes=args.num_classes, im_size=args.im_size, conv_num=args.conv_num, kernel_num=args.kernel_num).to(args.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.mapping_model.conv.parameters(), lr=1e-2)
        if args.method == 'ILT':
            self.optimizer = optim.Adam(self.mapping_model.parameters(), lr=1e-2)

        #FL本地模型
        if args.method == 'FL' or args.method == 'FedProx' or args.method == 'FedNova':
            self.local_net = get_network(args.model, args.channel, args.num_classes, args.im_size).to(args.device)
            self.optimizer = optim.Adam(self.local_net.parameters(), lr=5e-4)

        self.privacy_engine = PrivacyEngine()
        self.local_net, self.optimizer, self.train_loader = self.privacy_engine.make_private(
            module=self.local_net,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            noise_multiplier=1.0,
            max_grad_norm=1.0
        )

    def localDC(self, args):

        
        mapping_model_conv = copy.deepcopy(self.mapping_model).conv.to(args.device)
        for param in list(mapping_model_conv.parameters()):
            param.requires_grad = False

        #进行本地数据蒸馏


        print(f'\n================== Client {self.id} DC ==================\n')

        ''' organize the real dataset '''
        if self.images_all == None:
            images_all = []
            labels_all = []
            indices_class = [[] for c in range(args.num_classes)]

            images_all = [torch.unsqueeze(self.dst_train[i][0], dim=0) for i in range(len(self.dst_train))]
            labels_all = [self.dst_train[i][1] for i in range(len(self.dst_train))]
            for i, lab in enumerate(labels_all):
                indices_class[lab].append(i)
            self.images_all = torch.cat(images_all, dim=0).to(args.device)
            self.labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)
            self.indices_class = indices_class
            for c in range(args.num_classes):
                print('class c = %d: %d real images'%(c, len(indices_class[c])))

        def get_images(c, n): # get random n images from class c
            idx_shuffle = np.random.permutation(self.indices_class[c])[:n]
            return self.images_all[idx_shuffle]

        ''' initialize the synthetic data '''
        # if self.image_syn == None or 1:
        self.image_syn = torch.randn(size=(args.num_classes*args.ipc, 1 if args.mapping else args.channel, args.im_size[0], args.im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        self.label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(args.num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

        if args.init == 'real':
            print('initialize synthetic data from random real images')
            for c in range(args.num_classes):
                # np.random.seed(c)
                if args.mapping:
                    mapping_data = mapping_model_conv(get_images(c, args.ipc)).clone().detach().data
                    self.image_syn.data[c*args.ipc:(c+1)*args.ipc] = mapping_data.view(mapping_data.size(0), 1, args.im_size[0], args.im_size[1])
                else:
                    self.image_syn.data[c*args.ipc:(c+1)*args.ipc] = get_images(c, args.ipc).detach().data
        else:
            print('initialize synthetic data from random noise')


        ''' training '''
        optimizer_img = torch.optim.SGD([self.image_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        criterion = nn.CrossEntropyLoss().to(args.device)

        print('%s training begins'%get_time())

        for it in range(args.Iteration+1):

            ''' Train synthetic data '''
            net = get_network(args.model, 1 if args.mapping else args.channel, args.num_classes, args.im_size).to(args.device) # get a random model
            net.train()
            for param in list(net.parameters()):
                param.requires_grad = False

            embed = net.embed

            ''' update synthetic data '''
            loss = torch.tensor(0.0).to(args.device)

            for c in range(args.num_classes):
                img_real = get_images(c, args.batch_real)
                img_syn = self.image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, 1 if args.mapping else args.channel, args.im_size[0], args.im_size[1]))

                if args.dsa:
                    seed = int(time.time() * 1000) % 100000
                    img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                    img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                if args.mapping:
                    mapping_real = mapping_model_conv(img_real)
                    mapping_real = mapping_real.view(mapping_real.size(0), 1, args.im_size[0], args.im_size[1])
                    output_real = embed(mapping_real).detach()
                else:
                    output_real = embed(img_real).detach()
                output_syn = embed(img_syn)

                #MMD loss
                loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)

            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()

            #约束图片的数值
            # image_syn.data = torch.clamp(image_syn, 0.0, 1.0)

            if it%10 == 0:
                print('%s iter = %05d, loss = %.4f' % (get_time(), it, loss))

        return
    
    def localDC2(self, global_net_params, args):
        #加载全局模型
        global_net = get_network(args.model, 1 if args.mapping else args.channel, args.num_classes, args.im_size).to(args.device)
        global_net.load_state_dict(global_net_params)
        for param in list(global_net.parameters()):
            param.requires_grad = False

        #根据全局模型优化映射模块
        print(f'\n================== Client {self.id} mapping model training ==================\n')
        
        mapping_model_conv = self.mapping_model.conv

        # 训练过程
        num_epochs = args.epoch_mapping_model_train
        for epoch in range(num_epochs):
            num_iters = 0
            losses = 0
            mapping_model_conv.train()
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # 前向传播
                outputs = global_net(mapping_model_conv(images).reshape(images.size(0), 1, args.im_size[0], args.im_size[1]))
                loss = self.criterion(outputs, labels)

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses += loss.item()
                num_iters += 1
            losses /= num_iters
            print(f"Client {self.id}, epoch [{epoch+1}/{num_epochs}], loss: {losses:.6f}")

        # 测试过程
        mapping_model_conv.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = global_net(mapping_model_conv(images).reshape(images.size(0), 1, args.im_size[0], args.im_size[1]))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            acc = correct/total
            print(f'Client {self.id}, 本地测试集准确率: {100 * acc:.2f}%')


        #进行本地数据蒸馏
        print(f'\n================== Client {self.id} DC ==================\n')

        ''' organize the real dataset '''
        if self.images_all == None:
            images_all = []
            labels_all = []
            indices_class = [[] for c in range(args.num_classes)]

            images_all = [torch.unsqueeze(self.dst_train[i][0], dim=0) for i in range(len(self.dst_train))]
            labels_all = [self.dst_train[i][1] for i in range(len(self.dst_train))]
            for i, lab in enumerate(labels_all):
                indices_class[lab].append(i)
            self.images_all = torch.cat(images_all, dim=0).to(args.device)
            self.labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)
            self.indices_class = indices_class
            for c in range(args.num_classes):
                print('class c = %d: %d real images'%(c, len(indices_class[c])))
            #生成syn_img的序号
            self.syn_img_num = [0]*args.num_classes
            for c in range(args.num_classes):
                self.syn_img_num[c] = min(args.ipc, len(indices_class[c]))
            print(self.syn_img_num)
            self.syn_img_index = copy.deepcopy(self.syn_img_num)
            for c in range(1, args.num_classes):
                self.syn_img_index[c] += self.syn_img_index[c-1]
            print(self.syn_img_index)

        def get_images(c, n): # get random n images from class c
            idx_shuffle = np.random.permutation(self.indices_class[c])[:n]
            return self.images_all[idx_shuffle]

        ''' initialize the synthetic data '''
        self.image_syn = torch.randn(size=(self.syn_img_index[args.num_classes-1], 1 if args.mapping else args.channel, args.im_size[0], args.im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        label_syn_list = []
        for i in range(args.num_classes):
            if self.syn_img_num[i] != 0:
                label_syn_list.append(np.ones(self.syn_img_num[i])*i)
        self.label_syn = torch.tensor(np.concatenate(label_syn_list), dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

        if args.init == 'real':
            print('initialize synthetic data from random real images')
            for c in range(args.num_classes):
                # np.random.seed(c)
                if self.syn_img_num[c] == 0:
                    continue
                if args.mapping:
                    mapping_data = mapping_model_conv(get_images(c, self.syn_img_num[c])).clone().detach().data
                    self.image_syn.data[(0 if c==0 else self.syn_img_index[c-1]) : self.syn_img_index[c]] = mapping_data.reshape(mapping_data.size(0), 1, args.im_size[0], args.im_size[1])
                else:
                    self.image_syn.data[c*args.ipc:(c+1)*args.ipc] = get_images(c, args.ipc).detach().data
        else:
            print('initialize synthetic data from random noise')


        ''' training '''
        optimizer_img = torch.optim.SGD([self.image_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        criterion = nn.CrossEntropyLoss().to(args.device)

        print('%s training begins'%get_time())

        for it in range(args.Iteration+1):

            ''' Train synthetic data '''
            net = get_network(args.model, 1 if args.mapping else args.channel, args.num_classes, args.im_size).to(args.device) # get a random model
            
            #参考FedDM采样球形半径内的模型参数
            # if args.method == 'myFedDM':
            #     #获得全局模型半径为rho的随机模型参数
            #     rho = 5

            #     #生成噪音参数
            #     noise_params = {}
            #     for k, v in global_net_params.items():
            #         noise = torch.from_numpy(np.random.normal(loc=0., scale=1., size=v.shape)).to(args.device)
            #         noise_params[k] = noise
            #     noise_norm = torch.norm(torch.stack([torch.norm(v.detach(), 2).to(args.device) for _,v in noise_params.items()]), 2)
            #     threshold = min(1, rho/noise_norm)
            #     if threshold != 1:
            #         for k, v in noise_params.items():
            #             noise_params[k] = v*threshold
                
            #     #添加到全局模型中
            #     for k, v in noise_params.items():
            #         noise_params[k] += global_net_params[k]

            #     #随机模型加载参数
            #     net.load_state_dict(noise_params)

            net.train()
            for param in list(net.parameters()):
                param.requires_grad = False

            embed = net.embed

            ''' update synthetic data '''
            loss = torch.tensor(0.0).to(args.device)

            for c in range(args.num_classes):
                if len(self.indices_class[c]) == 0:
                    continue
                img_real = get_images(c, min(args.batch_real, len(self.indices_class[c])))
                # print(min(args.batch_real, len(indices_class[c])))
                img_syn = self.image_syn[(0 if c==0 else self.syn_img_index[c-1]) : self.syn_img_index[c]].reshape((self.syn_img_num[c], 1 if args.mapping else args.channel, args.im_size[0], args.im_size[1]))

                if args.dsa and len(img_syn)>1:
                    seed = int(time.time() * 1000) % 100000
                    img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                    # img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                if args.mapping:
                    mapping_real = mapping_model_conv(img_real)
                    mapping_real = mapping_real.reshape(mapping_real.size(0), 1, args.im_size[0], args.im_size[1])
                    output_real = embed(mapping_real).detach()
                else:
                    output_real = embed(img_real).detach()
                output_syn = embed(img_syn)

                #MMD loss
                loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)

            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()

            if it%10 == 0:
                print('%s iter = %05d, loss = %.4f' % (get_time(), it, loss))

        transform_to_pil = torchvision.transforms.ToPILImage()
        img = transform_to_pil(self.image_syn.data[0])
        img.save('img.png')
        

        return acc
    

    #FedDM方法
    def FedDM(self, global_net_params, args):
        
        #进行本地数据蒸馏
        print(f'\n================== Client {self.id} DC ==================\n')

        ''' organize the real dataset '''
        if self.images_all == None:
            images_all = []
            labels_all = []
            indices_class = [[] for c in range(args.num_classes)]

            images_all = [torch.unsqueeze(self.dst_train[i][0], dim=0) for i in range(len(self.dst_train))]
            labels_all = [self.dst_train[i][1] for i in range(len(self.dst_train))]
            for i, lab in enumerate(labels_all):
                indices_class[lab].append(i)
            self.images_all = torch.cat(images_all, dim=0).to(args.device)
            self.labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)
            self.indices_class = indices_class
            for c in range(args.num_classes):
                print('class c = %d: %d real images'%(c, len(indices_class[c])))
            #生成syn_img的序号
            self.syn_img_num = [0]*args.num_classes
            for c in range(args.num_classes):
                self.syn_img_num[c] = min(args.ipc, len(indices_class[c]))
            print(self.syn_img_num)
            self.syn_img_index = copy.deepcopy(self.syn_img_num)
            for c in range(1, args.num_classes):
                self.syn_img_index[c] += self.syn_img_index[c-1]
            print(self.syn_img_index)

        def get_images(c, n): # get random n images from class c
            idx_shuffle = np.random.permutation(self.indices_class[c])[:n]
            return self.images_all[idx_shuffle]

        ''' initialize the synthetic data '''
        self.image_syn = torch.randn(size=(self.syn_img_index[args.num_classes-1], args.channel, args.im_size[0], args.im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        label_syn_list = []
        for i in range(args.num_classes):
            if self.syn_img_num[i] != 0:
                label_syn_list.append(np.ones(self.syn_img_num[i])*i)
        self.label_syn = torch.tensor(np.concatenate(label_syn_list), dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

        if args.init == 'real':
            print('initialize synthetic data from random real images')
            for c in range(args.num_classes):
                # np.random.seed(c)
                if self.syn_img_num[c] == 0:
                    continue
    
                self.image_syn.data[(0 if c==0 else self.syn_img_index[c-1]) : self.syn_img_index[c]] = get_images(c, self.syn_img_num[c]).detach().data

        else:
            print('initialize synthetic data from random noise')


        ''' training '''
        optimizer_img = torch.optim.SGD([self.image_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        criterion = nn.CrossEntropyLoss().to(args.device)

        print('%s training begins'%get_time())

        for it in range(args.Iteration+1):

            ''' Train synthetic data '''
            net = get_network(args.model, 1 if args.mapping else args.channel, args.num_classes, args.im_size).to(args.device) # get a random model

            if args.method == 'FedDM':
                #获得全局模型半径为rho的随机模型参数
                rho = 5

                #生成噪音参数
                noise_params = {}
                for k, v in global_net_params.items():
                    noise = torch.from_numpy(np.random.normal(loc=0., scale=1., size=v.shape)).to(args.device)
                    noise_params[k] = noise
                noise_norm = torch.norm(torch.stack([torch.norm(v.detach(), 2).to(args.device) for _,v in noise_params.items()]), 2)
                threshold = min(1, rho/noise_norm)
                if threshold != 1:
                    for k, v in noise_params.items():
                        noise_params[k] = v*threshold
                
                #添加到全局模型中
                for k, v in noise_params.items():
                    noise_params[k] += global_net_params[k]

                #随机模型加载参数
                net.load_state_dict(noise_params)

            net.train()
            for param in list(net.parameters()):
                param.requires_grad = False

            embed = net.embed

            ''' update synthetic data '''
            loss = torch.tensor(0.0).to(args.device)

            for c in range(args.num_classes):
                if len(self.indices_class[c]) == 0:
                    continue
                img_real = get_images(c, min(args.batch_real, len(self.indices_class[c])))
                # print(min(args.batch_real, len(indices_class[c])))
                img_syn = self.image_syn[(0 if c==0 else self.syn_img_index[c-1]) : self.syn_img_index[c]].reshape((self.syn_img_num[c], args.channel, args.im_size[0], args.im_size[1]))

                if args.dsa and len(img_syn)>1:
                    seed = int(time.time() * 1000) % 100000
                    img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                    img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                output_real = embed(img_real).detach()
                output_syn = embed(img_syn)

                logits_real = net(img_real).detach()
                logits_syn = net(img_syn)

                #MMD loss
                loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2) + torch.sum((torch.mean(logits_real, dim=0) - torch.mean(logits_syn, dim=0))**2)

            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()

            if it%10 == 0:
                print('%s iter = %05d, loss = %.4f' % (get_time(), it, loss))

        return 
    
    def train_mapping_model(self):

        num_epochs = 20

        # 训练过程
        for epoch in range(num_epochs):
            num_iters = 0
            losses = 0
            self.mapping_model.train()
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # 前向传播
                outputs = self.mapping_model(images)
                loss = self.criterion(outputs, labels)

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses += loss.item()
                num_iters += 1
            losses /= num_iters
            print(f"Client {self.id}, epoch [{epoch+1}/{num_epochs}], loss: {losses:.4f}")

        # 测试过程
        self.mapping_model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.mapping_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f'Client {self.id}, mapping model 测试集准确率: {100 * correct / total:.2f}%')
        return

    def fl(self, global_net_p, args):
        #加载全局模型
        global_net_params={}
        for k,v in global_net_p.items():
            global_net_params['_module.'+k] = v
        self.local_net.load_state_dict(global_net_params)

        # 训练过程
        num_epochs = 1
        self.sgd_iters = 0
        
        self.local_net.train()
        for epoch in range(num_epochs):
            num_iters = 0
            losses = 0

            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                # 前向传播
                outputs = self.local_net(images)

                if args.method == 'FedProx':
                    mu = 0.01
                    proximal_term = 0.0
                    for k,w in self.local_net.named_parameters():
                        proximal_term += (w - global_net_params[k]).norm(2)
                    loss = self.criterion(outputs, labels) + (mu / 2) * proximal_term
                else:
                    loss = self.criterion(outputs, labels)

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses += loss.item()
                num_iters += 1
                self.sgd_iters += 1
            losses /= num_iters
            print(f"Client {self.id}, epoch [{epoch+1}/{num_epochs}], loss: {losses:.6f}")

            epsilon = self.privacy_engine.accountant.get_epsilon(delta=1e-5)
            print(f'epislon: {epsilon:.2f}')


        # 测试过程
        with torch.no_grad():
            correct = 0
            total = 0
            self.local_net.eval()
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.local_net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f'Client {self.id}, 本地测试集准确率: {100 * correct / total:.2f}%')
        
        #计算梯度
        grads = dict()
        for k,v in self.local_net.state_dict().items():
            grads[k] = v - global_net_params[k]
        if args.method == 'FedNova':
            for k,v in grads.items():
                grads[k] = v/self.sgd_iters

        return grads
    
    def individual_local_train(self, args):
        print(f'\n================== Client {self.id} individual local model training ==================\n')
        
        # 训练过程
        num_epochs = args.epoch_mapping_model_train
        for epoch in range(num_epochs):
            num_iters = 0
            losses = 0
            self.mapping_model.train()
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # 前向传播
                outputs = self.mapping_model(images)
                loss = self.criterion(outputs, labels)

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses += loss.item()
                num_iters += 1
            losses /= num_iters
            print(f"Client {self.id}, epoch [{epoch+1}/{num_epochs}], loss: {losses:.6f}")

        # 测试过程
        self.mapping_model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.mapping_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            acc = correct/total
            print(f'Client {self.id}, 本地测试集准确率: {100 * acc:.2f}%')

        return acc
    
class ClientsManager():
    def __init__(self, args) -> None:
        self.clients = []
        self.args = args
        
        #初始化所有的本地client
        self.clients_init()

    def clients_init(self):
        #分配客户端数据
        channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(self.args.Dataset)
        #合并训练集和测试集，重新划分
        dst_train = torch.utils.data.ConcatDataset([dst_train, dst_test])

        #iid场景
        if not self.args.noiid:
            # 计算每个分块的大小
            split_size = len(dst_train) // self.args.Clients_Num
            remainder = len(dst_train) % self.args.Clients_Num
            lengths = [split_size] * self.args.Clients_Num
            for i in range(remainder):
                lengths[i] += 1

            # 随机划分数据集
            splits = random_split(dst_train, lengths)

            for i in range(self.args.Clients_Num):
                client = Client( i+1, splits[i], self.args)
                self.clients.append(client)
        else:
            #noiid场景
            print('\n*** No-IID ***\n')
            dst_train = torch.load(f='./noiid_datasets/{}/u10c10-alpha{}-ratio100/train/train.pt'.format(self.args.Dataset, str(self.args.alpha)))
            dst_test = torch.load(f='./noiid_datasets/{}/u10c10-alpha{}-ratio100/test/test.pt'.format(self.args.Dataset, str(self.args.alpha)))
            for i in range(self.args.Clients_Num):
                username = dst_train['users'][i]
                userdata = dst_train['user_data'][username]
                X, y = userdata['x'], userdata['y']
                clientData = [(x, y) for x, y in zip(X, y)]

                client = Client( i+1, clientData, self.args)
                self.clients.append(client)

        
    

if __name__ == '__main__':
    
    pass