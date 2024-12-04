import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.transforms.transforms import Compose, ToTensor
from torchvision.utils import save_image
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms
import random
import numpy as np
import time
import os


def get_outputs_filename(subname):
    lt = list(time.localtime())
    lt = [str(x) for x in lt[:6]]
    fn = '-'.join(lt)
    fn = 'outputs/' + subname + '-' + fn
    if os.path.exists(fn):
        return fn
    else:
        os.makedirs(fn)
        os.makedirs(fn+'/models')
    return fn

def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    @staticmethod
    def _tensor_size(t):
        return t.size()[1]*t.size()[2]*t.size()[3]

def reconstruction_costs(gradients, input_gradient, cost_fn='l2'):

    indices = torch.arange(len(input_gradient))
    weights = input_gradient[0].new_ones(len(input_gradient))

    total_costs = 0
    for trial_gradient in gradients:
        pnorm = [0, 0]
        costs = 0

        for i in indices:
            if cost_fn == 'l2':
                costs += ((trial_gradient[i] - input_gradient[i]).pow(2)).sum() * weights[i]
            elif cost_fn == 'l1':
                costs += ((trial_gradient[i] - input_gradient[i]).abs()).sum() * weights[i]
            elif cost_fn == 'max':
                costs += ((trial_gradient[i] - input_gradient[i]).abs()).max() * weights[i]
            elif cost_fn == 'sim':
                costs -= (trial_gradient[i] * input_gradient[i]).sum() * weights[i]
                pnorm[0] += trial_gradient[i].pow(2).sum() * weights[i]
                pnorm[1] += input_gradient[i].pow(2).sum() * weights[i]

        if cost_fn == 'sim':
            costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()

        # Accumulate final costs
        total_costs += costs
    return total_costs / len(gradients)


def geDLG(netTE, gt_data, gt_label, gt_embed, original_dy_dx=[], opt='Adam', filename=None, r=1000, lr=1e-3, loss_factors=[1e-2, 1, 1e-2], device='cpu'):

    b1, b2, b3 = loss_factors
    print("loss factors: %f  %f  %f" % (b1, b2, b3))

    criterion = F.cross_entropy
    tv_loss = TVLoss()

    netTE.eval()
    embd = netTE.embed

    # generate dummy data and label
    dummy_data = torch.randn(*gt_data.size()).to(device).requires_grad_(True)

    if opt == 'LBFGS':
        optimizer = torch.optim.LBFGS([dummy_data])
    elif opt == 'RMS':
        optimizer = torch.optim.RMSprop([dummy_data], lr=lr, momentum=0.99)
    elif opt == 'Adam':
        optimizer = torch.optim.Adam([dummy_data], lr=lr)
    elif opt == 'SGD':
        optimizer = torch.optim.SGD([dummy_data], lr=lr, momentum=0.99)

    scheduler = MultiStepLR(optimizer, milestones=[r // 2.667, r // 1.6, r // 1.142], gamma=0.1)

    psnr_max = 0
    record_loss = []
    record_psnr = []
    for iters in range(r):
        optimizer.zero_grad()
        netTE.zero_grad()

        dummy_pred = netTE(dummy_data) 
        dummy_loss = criterion(dummy_pred, gt_label)
        dummy_dy_dx = torch.autograd.grad(dummy_loss, netTE.parameters(), create_graph=True)
        #the gradient matching loss
        loss1 = reconstruction_costs([dummy_dy_dx], original_dy_dx, cost_fn='sim')  #l2  sim

        dummy_f = embd(dummy_data) 
        #the feature matching loss
        loss2 = reconstruction_costs([dummy_f], gt_embed, cost_fn='sim')  #l2  sim
        #the TV loss
        loss3 = tv_loss(dummy_data)

        loss = b1*loss1 + b2*loss2 +b3*loss3

        loss.backward()

        optimizer.step()
        scheduler.step()

        dummy_data.data = torch.clamp(dummy_data, 0, 1)

        if (iters+1) % 50 == 0: 
            mse = torch.mean((gt_data.detach()-dummy_data.detach())**2, dim=(1,2,3))
            psnr = 10*torch.log10(1/mse)
            psnr_mean = torch.mean(psnr)
            print('rounds: [%d|%d]  ||  lr: %.4f  ||  loss: %.6f  ||  PSNR: %.6f' %(iters+1, r, scheduler.get_last_lr()[0], loss.item(), psnr_mean))

            figs = []
            for i in range(len(gt_data)):
                figs.append(gt_data[i])
                figs.append(dummy_data[i])
            figs = torch.stack(figs).float()
            if psnr_max<=psnr_mean.item():
                save_image(figs.clone().detach().cpu(), filename+'-original&recoveredImages.png' )
                # save_image(dummy_data[:16].clone().detach().cpu(), filename+'rdataB16.png', nrow=4)
            record_loss.append(loss.item())
            record_psnr.append(psnr_mean.item())
            psnr_max = max(psnr_mean.item(), psnr_max)

    return psnr_max


#compute the similarities between recovered inputs and the original inputs to 1st FC layer
def inverse_scores(data, fdata):
    data = data.clone().detach()
    fdata = fdata.clone().detach()
    data = data.view(data.shape[0], -1)
    fdata = fdata.view(fdata.shape[0], -1)

    d_l2 = torch.cdist(data.unsqueeze(0), fdata.unsqueeze(0), 2)
    d_l2_min, index = torch.min(d_l2, 2)

    return d_l2_min, index

def PSNRs(data, fdata):
    diff = data - fdata
    diff2 = diff*diff
    if len(data.shape)==2:
        mse = torch.mean(diff2)
    else:
        h = data.shape[2]
        mse = torch.sum(diff2, dim=[2, 3])/(h*h)
        mse = torch.mean(mse, dim=1)
    psnr = 10*torch.log10(1/mse)
    return psnr, mse


#显示fc层的输入
def GradientInversion(data, label, netTE, criterion, filename, c, h, w, args, SAVE_FIG=False, device='cpu'):

    original_data = data.detach().clone()

    #compute the gradient
    if args.CNN:
        #compute the input data to the 1st FC layer, which is the features after the convolution layers in CNNs
        netTE.eval()
        data = netTE.embed(data)
        #compute the gradient of the CNNs
        netTE.train()
        pred = netTE(original_data)
        y = criterion(pred, label)
        dy_dx = torch.autograd.grad(y, netTE.parameters())
    else:
        #compute the gradient of the FCNNs
        pred = netTE(data)
        y = criterion(pred, label)
        dy_dx = torch.autograd.grad(y, netTE.parameters())

    original_dy_dx = []
    for g in dy_dx:
        original_dy_dx.append(g.detach().clone())

    #obtain the gradients of the 1st FC layer in models
    if args.CNN:
        original_dy_dx_fc = original_dy_dx[-4:-2]
    else:
        original_dy_dx_fc = original_dy_dx[0:2]

    #obtain all inputs to 1st FC with analytic method
    xinall = []
    for i in range(len(original_dy_dx_fc[1])):
        if original_dy_dx_fc[1][i] == 0:
            xin = torch.zeros(original_dy_dx_fc[0][i].shape).to(device)
        else:
            xin = original_dy_dx_fc[0][i]/original_dy_dx_fc[1][i]
        if args.CNN == False:
            xin = xin.view(c,h,w)
            xin = xin[:3, :, :]
        xinall.append(xin)

    xinall = torch.stack(xinall)
    xinall = xinall.float()

    #save the recovered data from all neurons in the 1st FC layer
    if SAVE_FIG and not args.CNN:
        save_image(xinall.cpu().detach(), filename+'-recoveredImagesOfAllNeurons.png')

    #compute the similarities between recovered inputs and the original inputs to 1st FC layer
    dl2_min, min_index = inverse_scores(data, xinall)

    #xinall2 collects original inputs and recovered inputs that has the smallest difference with original inputs
    xinall2 = []
    psnrs = []
    for i in range(len(data)):
        xinall2.append(data[i])
        xinall2.append(xinall[min_index[0][i]])
        psnr, mse = PSNRs(torch.unsqueeze(data[i], 0) , torch.unsqueeze(xinall[min_index[0][i]], 0))
        psnrs.append(psnr.cpu())

    xinall2 = torch.stack(xinall2)
    psnrs = torch.stack(psnrs)
    psnrs[psnrs>100] = 100

    if SAVE_FIG and not args.CNN:
        save_image(xinall2.cpu().detach(), filename+'-original&recoveredImages.png')

    #exploit DLG to recover original images from recovered features after convolution layers
    if args.CNN:
        embed_features = torch.stack([xinall2[2*i+1].detach() for i in range(len(original_data))]) 
        psnrs = geDLG(netTE, original_data.detach(), label, embed_features, original_dy_dx, 'Adam', filename, 3000, 1e-1, [args.b1, args.b2, args.b3], device)
        psnrs = torch.Tensor([psnrs])

    

    score1 = int(torch.sum(psnrs>=20))/len(data)
    score2 = int(torch.sum(psnrs>=30))/len(data)
    score3 = int(torch.sum(psnrs>=40))/len(data)
    score4 = int(torch.sum(psnrs>=45))/len(data)

    psnr_mean = torch.mean(psnrs).detach().numpy()

    return [score1, score2, score3, score4, psnr_mean]


def get_maxindex(output, k, t, device):
    avgt = torch.mean(t)

    output = output.clone().detach()
    maxindex = torch.Tensor([]).long().to(device)
    for i in range(len(output)):
        outputi = output[i]
        outputi[maxindex] = -1
        outputi[t>avgt] = -1
        _, index = torch.topk(outputi, k)
        maxindex = torch.concat((maxindex, index))
    t[maxindex] += 1
    res = []
    for i in range(k):
        item = [maxindex[k*j+i] for j in range(len(output))]
        res.append(torch.Tensor(item).long().to(device))
    return res

#the training loss for malicious parameters
def MPloss(output, k, t, device):
    loss = 0

    lossFun = nn.CrossEntropyLoss()
    #choose neurons to be SDANs according to the outputs of the FC layer
    maxindex = get_maxindex(output, k, t, device)
    for i in range(k):
        loss += lossFun(output, maxindex[i])
    loss /= k

    return loss

#generate malicious parameters with Gaussian distribution for each neuron
def randomMP(f_dim, b, s):
    randomIndex = np.random.choice(f_dim, f_dim, replace=False)
    P = randomIndex[:f_dim//2]
    N = randomIndex[f_dim//2:]
    zN = np.random.normal(loc=0, scale=b, size=f_dim//2)
    zP = zN.copy()*s*(-1)
    np.random.shuffle(zN)
    np.random.shuffle(zP)
    rMP = np.zeros(f_dim)
    for i in range(f_dim//2):
        rMP[N[i]]=zN[i]
        rMP[P[i]]=zP[i]
    return rMP

#generate malicious parameters with Gaussian distribution for one FC layer
def RMPinit(b, s, f_dim, L, net):
    weight = []
    for i in range(L):
        rmp = randomMP(f_dim, b, s)
        weight.append(rmp)
    weight = torch.Tensor(np.array(weight))
    bias = torch.zeros(L)
    p = net.state_dict()
    p['linear0.weight'] = weight
    p['linear0.bias'] = bias
    net.load_state_dict(p)

def get_dataset(args):
    data_root = "~/.torch"
    transform = Compose([
        ToTensor()
        ])

    f_dim = 0
    nc = 0

    if args.DATA == 'MNIST':
        trainset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
        testset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
        nc = 10
        c, h, w = 1, 28, 28
        f_dim = 512 if args.CNN else c*h*w

    elif args.DATA == 'CIFAR10':
        trainset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
        nc = 10
        c, h, w = 3, 32, 32
        f_dim = 2048 if args.CNN else c*h*w

    elif args.DATA == 'CIFAR100':
        trainset = datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform)
        testset = datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform)
        nc = 100
        c, h, w = 3, 32, 32
        f_dim = 2048 if args.CNN else c*h*w

    elif args.DATA == 'FashionMNIST':
        trainset = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=transform)
        testset = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=transform)
        nc = 10
        c, h, w = 1, 28, 28
        f_dim = 512 if args.CNN else c*h*w

    elif args.DATA == 'FaceScrub':
        transform = Compose([
        transforms.Resize(64),
        ToTensor()
        ])
        data = datasets.ImageFolder(root=data_root+'/FaceScrub', transform=transform)
        trainset, testset = torch.utils.data.random_split(data, [int(len(data)*0.7), len(data)-int(len(data)*0.7)])
        nc = 530
        c, h, w = 3, 64, 64
        f_dim = 8192 if args.CNN else c*h*w

    elif args.DATA == 'TinyImagenet':
        transform = Compose([
        transforms.Resize(64),
        ToTensor()
        ])
        trainset = datasets.ImageFolder(root=data_root+'/tiny-imagenet-200/train', transform=transform)
        testset = datasets.ImageFolder(root=data_root+'/tiny-imagenet-200/test', transform=transform)
        nc = 200
        c, h, w = 3, 64, 64
        f_dim = 8192 if args.CNN else c*h*w

    return trainset, testset, nc, c, h, w, f_dim