import torchvision
import torchvision.transforms as transforms
import random
import torch
# from torchtext import data

class DataSplit():
    def __init__(self, dataSetName, isIID, clientsNum, args) -> None:
        self.dataSetName = dataSetName
        self.isIID = isIID
        self.clientsNum = clientsNum
        self.trainset = None
        self.testset = None
        self.publicdata_topk = None
        self.args = args
        

        self.prepareData()
        

    def prepareData(self):
        if self.dataSetName == 'mnist':
            # transform = transforms.Compose([
            #     transforms.Resize((224, 224)),
            #     transforms.Grayscale(3),
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
            # ])
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307), (0.3081))
            ])
            self.trainset = list(torchvision.datasets.MNIST(root='~/.torch', train=True, download=True, transform=transform))
            self.testset = torchvision.datasets.MNIST(root='~/.torch', train=False, download=True, transform=transform)
            print('# mnist数据加载成功 #')
            if not self.isIID:
                self.trainset.sort(key=lambda x:x[1])
            # else:
            #     random.shuffle(self.trainset)

        elif self.dataSetName == 'fmnist':
            transform = transforms.Compose([transforms.ToTensor()])
            self.trainset = list(torchvision.datasets.FashionMNIST(root='~/.torch', train=True, download=True, transform=transform))
            self.testset = torchvision.datasets.FashionMNIST(root='~/.torch', train=False, download=True, transform=transform)
            print('# fmnist数据加载成功 #')
            self.publicdata_topk = random.sample(self.trainset, 1000)
            if not self.isIID:
                self.trainset.sort(key=lambda x:x[1])
        
        elif self.dataSetName == 'emnist':
            transform = transforms.Compose([transforms.ToTensor()])
            self.trainset = list(torchvision.datasets.EMNIST(root='~/.torch', train=True, download=True, transform=transform, split='byclass'))
            self.testset = torchvision.datasets.EMNIST(root='~/.torch', train=False, download=True, transform=transform, split='byclass')
            print('# fmnist数据加载成功 #')
            self.publicdata_topk = random.sample(self.trainset, 1000)
            if not self.isIID:
                self.trainset.sort(key=lambda x:x[1])

        elif self.dataSetName == 'cifar10':
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
            self.trainset = list(torchvision.datasets.CIFAR10(root='~/.torch', train=True, download=True, transform=transform))
            self.testset = torchvision.datasets.CIFAR10(root='~/.torch', train=False, download=True, transform=transform)
            print('# cifar10数据加载成功 #')
            self.publicdata_topk = random.sample(self.trainset, 1000)
            if not self.isIID:
                #读取生成的noneiid数据集
                self.trainset = torch.load(f='noiid_data/cifar10/u100c10-alpha{}-ratio100/train/train.pt'.format(str(self.args.alpha)))

        elif self.dataSetName == 'svhn':
            transform = transforms.Compose([transforms.ToTensor()])
            self.trainset = list(torchvision.datasets.SVHN(root='~/.torch', download=True, transform=transform, split='train'))
            self.testset = torchvision.datasets.SVHN(root='~/.torch', download=True, transform=transform, split='test')
            # extraset = torchvision.datasets.SVHN(root='~/.torch', download=True, transform=transform, split='extra')

            print('# SVHN数据加载成功 #')
            self.publicdata_topk = random.sample(self.trainset, 1000)
            if not self.isIID:
                #读取生成的noneiid数据集
                print('读取noiid数据集')
                self.trainset = torch.load(f='noiid_data/SVHN/u100c10-alpha{}-ratio100/train/train.pt'.format(str(self.args.alpha)))

        elif self.dataSetName == 'TWEET':
            #定义加载参数
            LABEL = data.LabelField()
            TWEET = data.Field(lower=True)
            fields = [('score', None), ('id', None), ('data', None), ('query', None), ('name', None), ('tweet', TWEET), ('category', None), ('label', LABEL)]
            #加载twitter数据集
            twitterDataset = data.TabularDataset(
                path='~/.torch/TWITTER/training-processed.csv',
                format='CSV',
                fields=fields,
                skip_header=False,
            )
            #划分数据集
            self.trainset, self.testset, self.publicdata_topk = twitterDataset.split(split_ratio=[0.8, 0.1, 0.1])
            #生成词汇表
            vocab_size = 20000
            TWEET.build_vocab(self.trainset, max_size=vocab_size)
            LABEL.build_vocab(self.trainset)

            val_iter, test_iter = data.BucketIterator.splits(
                (   self.publicdata_topk, self.testset),
                    batch_size=64,
                    device=self.args.device,
                    sort_within_batch=True,
                    sort_key=lambda x:len(x.tweet)
                )
            self.testset = test_iter
            self.publicdata_topk = val_iter

            #TWEET划分各个客户端数据集
            clients_datasets = []
            train_main = self.trainset
            for i in range(99):
                percent = 1/(100-i)
                train_main, train_sub = train_main.split(split_ratio=[1-percent, percent])
                clients_datasets.append(train_sub)
            clients_datasets.append(train_main)
            self.trainset = clients_datasets

        else:
            raise ValueError("### dataSetName must be 'mnist' or 'cifar10' ###")
        print('训练数据总数： %d' % (len(self.trainset)))
        print('测试数据综述： %d' % (len(self.testset)))


    def getTrainSet(self, clientNum):
        if clientNum>self.clientsNum:
            raise ValueError('### clientNum必须小于等于clientsNum ###')
        
        if self.dataSetName == 'TWEET':
            return self.trainset[clientNum-1]

        if self.isIID:
            #将所有训练数据分成2*clientsNum的片段，每个client得到2个片段
            fractionNum = int(len(self.trainset)/(2*self.clientsNum))
            clientData = self.trainset[(clientNum-1)*fractionNum:clientNum*fractionNum] + self.trainset[(clientNum-1+self.clientsNum)*fractionNum:(clientNum+self.clientsNum)*fractionNum]
        else:
            #noneiid
            clientData = []
            username = self.trainset['users'][clientNum-1]
            userdata = self.trainset['user_data'][username]
            X, y = userdata['x'], userdata['y']
            clientData += [(x, y) for x, y in zip(X, y)]
        
        return clientData

