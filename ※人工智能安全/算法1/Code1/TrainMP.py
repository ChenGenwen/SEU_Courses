import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms.transforms import *
from utils import MPloss, get_outputs_filename, RMPinit, get_dataset
from networks import ConvNet, construct_model
from torch.utils.tensorboard import SummaryWriter
import argparse

parser = argparse.ArgumentParser(description='Parameter Processing')
parser.add_argument('--BATCHSIZE', type=int, default=64, help='batchsize')
parser.add_argument('--DATA', type=str, default='CIFAR100', help='dataset')
parser.add_argument('--P', type=float, default=1, help='the percentage of data used to train MP')
parser.add_argument('--CNN', action='store_true', help='use CNN model')
parser.add_argument('--FCNum', type=int, default=1024, help='the number of neurons in 1st FC layer')
parser.add_argument('--K', type=int, default=4, help='the number of single-data-activated neurons for each data point')
parser.add_argument('--EPOCH', type=int, default=100, help='the learning epochs')
parser.add_argument('--LR', type=float, default=1e-3, help='the learning rate')
args = parser.parse_args()


device = "cpu"
if torch.cuda.is_available():
    device = "cuda:1"
print("Running on %s" % device)


#create output filename
outputs_location = get_outputs_filename('MP-{}-K{}-B{}-{}'.format(args.DATA, args.K, args.BATCHSIZE, 'CNN' if args.CNN else 'FCNN'))


#init tensorboard
writer = SummaryWriter(outputs_location+'/logs')

#nc:number of classes, c:channels, f_dim:the dimensions of inputs to the 1st FC layer
trainset, testset, nc, c, h, w, f_dim = get_dataset(args)
#sample part of the training set
S1 = torch.utils.data.sampler.SubsetRandomSampler(np.random.choice(range(len(testset)), int(args.P*len(testset)), replace=False))
#the trainset is used to validate the malicious parameters
trainLoader = DataLoader(trainset, batch_size=args.BATCHSIZE, shuffle=True, num_workers=16, pin_memory=True)
#the testset is used to train malicious parameters
testLoader = DataLoader(testset, batch_size=args.BATCHSIZE, sampler=S1, num_workers=16, pin_memory=True)



if args.CNN:
    netTR = construct_model('OneMLP_train', fc1in_width=f_dim, fc1out_width=args.FCNum).to(device)
    netTE = construct_model('OneMLP_test', fc1in_width=f_dim, fc1out_width=args.FCNum).to(device)
    netemb = ConvNet(channel=c, num_classes=nc, net_width=128, net_depth=3, net_act='relu', net_norm='instancenorm', net_pooling='avgpooling', im_size=(h, w)).to(device)
    #Here we keep the parameters of convolution layers unchanged.
    if c==3:
        freezedCNN_p = torch.load('./model-saved/ConvNet/ConvNet3-rinit.pth', map_location=device)
    elif c==1:
        ofreezedCNN_pip = torch.load('./model-saved/ConvNet/ConvNet1-rinit.pth')
    netemb_p = netemb.state_dict()
    embed_p = {k:v for k, v in freezedCNN_p.items() if k[0:8]=='features'}
    netemb_p.update(embed_p)
    netemb.load_state_dict(netemb_p)
else:
    netTR = construct_model('OneMLP_train', fc1in_width=f_dim, fc1out_width=args.FCNum).to(device)
    netTE = construct_model('OneMLP_test', fc1in_width=f_dim, fc1out_width=args.FCNum).to(device)


optimizer = torch.optim.Adam(netTR.parameters(), lr=args.LR)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [args.EPOCH // 2.667, args.EPOCH // 1.6, args.EPOCH // 1.142], 0.1)


#init the malicious parameters of the first FC layer with Gaussian distribution
RMPinit(2, 0.97, f_dim, args.FCNum, netTR)

netTE.load_state_dict(netTR.state_dict())


#train the malicious parameters
for iter in range(args.EPOCH):
    running_loss = 0

    #the table used to count the neurons that have been selected as SDANs
    NeuralTable = torch.zeros(args.FCNum).to(device)

    for i, data in enumerate(testLoader):
        optimizer.zero_grad()

        img, label = data[0].to(device), data[1].to(device)
        #if the model is CNN, we use the features after the convolution layers to train malicious parameters
        if args.CNN:
            netemb.eval()
            img = netemb.embed(img)
        output = netTR(img)
        
        loss = MPloss(output, args.K, NeuralTable, device)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
    scheduler.step()

    running_loss /= len(testLoader)
    writer.add_scalar('loss', running_loss, iter)
    print('epoch %d: loss  %.6f' % (iter+1, running_loss))

    #save the malicious parameters
    SAVE_POINT = (iter+1)%50==0
    if SAVE_POINT:
        torch.save(netTR.state_dict(), outputs_location+'/models/netTE-e{}.pth'.format(iter+1)) 



    