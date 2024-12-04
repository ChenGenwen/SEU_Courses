import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from utils import GradientInversion, get_outputs_filename, setup_seed, get_dataset
import argparse
from networks import ConvNet, MLP
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
print("Running on %s" % device)

setup_seed(0)

parser = argparse.ArgumentParser(description='Parameter Processing')
parser.add_argument('--BATCHSIZE', type=int, default=64, help='batchsize')
parser.add_argument('--DATA', type=str, default='CIFAR100', help='dataset')
parser.add_argument('--b1', type=float, default=1e-3, help='gradient matching loss factor')
parser.add_argument('--b2', type=float, default=1, help='features matching loss factortaset')
parser.add_argument('--b3', type=float, default=1e-1, help='TV loss factor')
parser.add_argument('--noPOISON', action='store_true', help='whether poison the FC layer')
parser.add_argument('--CNN', action='store_true', help='use CNN model')
parser.add_argument('--TestNum', type=int, default=10, help='the number of batches to test')
parser.add_argument('--dropout_p', type=float, default=0.0, help='the probability of dropout layer')
parser.add_argument('--FCNum', type=int, default=1024, help='the number of neurons in 1st FC layer')
parser.add_argument('--MPfile', type=str, default="MP-saved/CNN-CIFAR100/FC1_MP.pth", help='malicious parameters filename')
args = parser.parse_args()


#create output filename
outputs_location = get_outputs_filename('MPtest-{}-B{}-{}'.format(args.DATA, args.BATCHSIZE, 'CNN' if args.CNN else 'FCNN'))

#初始化tensorboard
writer = SummaryWriter(outputs_location+'/logs')


trainset, testset, nc, c, h, w, f_dim = get_dataset(args)

lossFun = nn.CrossEntropyLoss()
trainLoader = DataLoader(trainset, batch_size=args.BATCHSIZE, shuffle=True, num_workers=0)
testLoader = DataLoader(testset, batch_size=args.BATCHSIZE, shuffle=True, num_workers=0)

# fc1_MPfile = 'MP-saved/CNN-CIFAR100/FC1_MP.pth'
# fc1_MPfile = 'MP-saved/CNN-FashionMNIST/FC1_MP.pth'
# fc1_MPfile = 'MP-saved/FCNN-CIFAR100/FC1_MP.pth'
# fc1_MPfile = 'MP-saved/FCNN-FashionMNIST/FC1_MP.pth'
fc1_MPfile = args.MPfile

if args.CNN:
    #create the CNN model
    netTE = ConvNet(channel=c, num_classes=nc, net_width=128, net_depth=3, net_act='relu', net_norm='instancenorm', net_pooling='avgpooling', im_size=(h, w)).to(device)

    #load the freezed parameters of the convolutions in CNN
    if c==3:
        freezedCNN_p = torch.load('./model-saved/ConvNet/ConvNet3-rinit.pth', map_location=device)
    elif c==1:
        freezedCNN_p = torch.load('./model-saved/ConvNet/ConvNet1-rinit.pth', map_location=device)
    netTE_p = netTE.state_dict()
    embed_p = {k:v for k, v in freezedCNN_p.items() if k[0:8]=='features'}
    netTE_p.update(embed_p)

    #load the trained malicious parameters of the first FC layer in CNN
    if not args.noPOISON:
        fc1_mp = torch.load(fc1_MPfile, map_location=device)
        netTE_p['classifier.0.weight']=fc1_mp['linear0.weight']
        netTE_p['classifier.0.bias']=fc1_mp['linear0.bias']

    netTE.load_state_dict(netTE_p)
else:
    netemb = None
    #create the FCNN model
    netTE = MLP(fc1in_width=f_dim, fc1out_width=args.FCNum, num_classes=nc, act_layer='ReLU', dropout_p=args.dropout_p).to(device)
    #load the malicious parameters of the first FC layer
    if not args.noPOISON:
        pn = netTE.state_dict()
        mp = torch.load(fc1_MPfile, map_location=device)
        pn['model.linear0.weight']=mp['linear0.weight']
        pn['model.linear0.bias']=mp['linear0.bias']
        netTE.load_state_dict(pn)



scores = []
for i, data in enumerate(trainLoader):
    print('Gradient Inversion Test, {}th batch data'.format(i+1))

    img, label = data[0].to(device), data[1].to(device)

    SAVE_FIG = True
    
    score = GradientInversion(img, label, netTE, lossFun, outputs_location+'/{}thBatch'.format(i+1), c, h, w, args, SAVE_FIG, device)
    scores.append(score)

    if i>=args.TestNum-1:
        break

avgscore = np.mean(scores, 0)

print("{}, Batchszie {}".format(args.DATA, args.BATCHSIZE))
print('the percentage of recovered images with PSNR over 40dB: %.2f%%' % (avgscore[2]*100))
print('the percentage of recovered images with PSNR over 30dB: %.2f%%' % (avgscore[1]*100))
print('the percentage of recovered images with PSNR over 20dB: %.2f%%' % (avgscore[0]*100))
print('average PSNR : %.6f' % (avgscore[4]))
stdscore = np.std(scores, 0)
print(stdscore)






