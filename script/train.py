import os
import numpy as np
import argparse
import time
import copy
import sys
import pandas as pd
from datetime import datetime 

import torch
import torch.nn.functional as F
torch.manual_seed(0)

package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(package_dir)
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from datas.CMCDatasets import CMCDatasets
from Net.model import Model
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from sklearn.metrics import r2_score
import random

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
random_seed=42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
# torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

EPS = 1e-10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=64, help='size of the batches')
parser.add_argument('--cdim', type=int, default=256, help='dimension of cluster feature')
parser.add_argument('--ncluster', type=int, default=14, help='number of cluster')
parser.add_argument('--lr', type = float, default=0.001, help='learning rate')
parser.add_argument('--stepsize', type=int, default=50, help='scheduler step size')
parser.add_argument('--gamma', type=float, default=0.4, help='scheduler shrinking rate')
parser.add_argument('--hiddendim', type=int, default=128, help='hiddeb dim')
parser.add_argument('--dropout', type=int, default=0.5, help='dropout probab')
parser.add_argument('--nlayers', type=int, default=1, help='number of layers')
parser.add_argument('--roi', type=int, default=374, help='number of roi')
parser.add_argument('--optim', type=str, default='Adam', help='optimization method: SGD, Adam')
parser.add_argument('--save_path', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)),'model'), help='path to save model')
parser.add_argument('--model', type=str, default='GAT', help='model type')
opt = parser.parse_args()

if not os.path.exists(opt.save_path):
    os.makedirs(opt.save_path)

#################### Parameter Initialization #######################

opt_method = opt.optim
num_epoch = opt.n_epochs
roi = opt.roi
hiddendim = opt.hiddendim
dropout = opt.dropout
nlayers = opt.nlayers
modelType = opt.model
cdim = opt.cdim
ncluster = opt.ncluster
batchSize = opt.batchSize
lr=opt.lr
save_path=opt.save_path

################## Define Dataloader ##################################

datas=torch.load('/private/jwyoun/01-PACC/02-experiments/4-GNN/datas/graphDatas2.pt')
dataset = CMCDatasets(datas)

total_size = len(dataset)
train_size = int(0.8 * total_size)  
test_size = total_size - train_size  

train_dataset, val_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset,batch_size=batchSize, shuffle= True)
val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False)

############### Define Graph Deep Learning Network ##########################
model = Model(roi,hiddendim,dropout,nlayers,modelType,cdim,ncluster).to(device)
# print(model)

if opt_method == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr= lr, )
elif opt_method == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr =lr, momentum = 0.9, nesterov = True)

scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                    first_cycle_steps=50,
                                    cycle_mult=2.,
                                    max_lr=0.01,
                                    min_lr=0.,
                                    warmup_steps=5,
                                    gamma=0.5)

###################### Network Training Function#####################################
def train(loader):
    print('train...........')

    for param_group in optimizer.param_groups:
        print("LR", param_group['lr'])
    model.train()

    loss_all = 0
    step = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x,data.edge_index,data)
        cluster_loss=0
        loss = F.mse_loss(output*10, data.y*10)+cluster_loss

        step = step + 1

        loss.backward()
        loss_all += loss.item()
        optimizer.step()

    scheduler.step()
    return loss_all / len(loader.dataset)

###################### Network Testing Function#####################################
def test_r2(loader):
    model.eval()
    test_predictions = []
    test_targets = []
    for data in loader:
        data = data.to(device)
        output = model(data.x,data.edge_index,data)
        test_predictions.extend(output.detach().cpu().numpy())
        test_targets.extend(data.y.detach().cpu().numpy())
    test_r2 = np.round(r2_score(test_targets, test_predictions),4)
    return test_r2

def test_loss(loader):
    model.eval()
    loss_all = 0
    for data in loader:
        data = data.to(device)
        output= model(data.x,data.edge_index,data)
        cluster_loss=0
        loss = F.mse_loss(output*10, data.y*10)+cluster_loss
        loss_all += loss.item() 
    return loss_all / len(loader.dataset)

def test_rmse(loader):
    model.eval()
    test_predictions = []
    test_targets = []
    for data in loader:
        data = data.to(device)
        output = model(data.x,data.edge_index,data)
        test_predictions.extend(output.detach().cpu().numpy())
        test_targets.extend(data.y.detach().cpu().numpy())
    test_predictions = np.array(test_predictions)
    test_targets = np.array(test_targets)
    mse = np.mean((test_predictions - test_targets) ** 2)
    rmse = np.round(np.sqrt(mse),4)
    return rmse

#######################################################################################
############################   Model Training #########################################
#######################################################################################
for epoch in range(0, num_epoch):
    since  = time.time()
    tr_loss= train(train_loader)
    model.eval()
    tr_loss=test_loss(train_loader)
    tr_acc = test_r2(train_loader)
    val_acc = test_r2(val_loader)
    val_loss = test_loss(val_loader)
    time_elapsed = time.time() - since
    print('*====**')
    # print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(' Epoch: {:03d}, Train Loss: {:.7f}, '
        'Train R^2 socre: {:.7f}, Val Loss: {:.7f}, Val Acc: {:.7f}'.format(epoch, tr_loss,
                                                    tr_acc, val_loss, val_acc))

model.eval()
tr_acc = test_r2(train_loader)
tr_rmse=test_rmse(train_loader)
te_acc = test_r2(val_loader)
te_rmse=test_rmse(val_loader)

print('''#######################################################################################
######################### Train Results ######################################
#######################################################################################
        ''')
print("Train R^2 score: {:.7f}, Test  R^2 score: {:.7f} ".format(tr_acc, te_acc))
print("Train rmse: {:.7f}, Test  rmse: {:.7f} ".format(tr_rmse, te_rmse))


print("saving model")
best_model_wts = copy.deepcopy(model.state_dict())
torch.save(best_model_wts, os.path.join(save_path,f"{modelType}_model("+datetime.today().strftime("%Y%m%d")+")"))