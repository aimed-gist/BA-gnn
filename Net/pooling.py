from typing import Optional

from torch import Tensor

from torch_geometric.utils import scatter
import torch_geometric.typing
from torch_geometric import is_compiling, warnings
from torch_geometric.typing import torch_scatter
from torch_geometric.utils.functions import cumsum
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class global_cluster_pool(nn.Module):
    def __init__(self,roi,hidden_dim,cdim,ncluster=14,dropout=0.5):
        super(global_cluster_pool, self).__init__()
        self.c=ncluster
        self.cluster_layers1 = nn.ModuleList()   
        self.cluster_layers2 = nn.ModuleList()  
        self.relu = nn.LeakyReLU()
        self.droupout=dropout
        self.r=roi
        self.linear_model=nn.Linear(self.r, self.c, bias=False)
        for i in range(self.c):
            layer1 = nn.Linear(hidden_dim, hidden_dim)
            bn1 = nn.BatchNorm1d(hidden_dim)  
            layer2 = nn.Linear(hidden_dim, cdim)
            bn2 = nn.BatchNorm1d(cdim) 

            torch.nn.init.kaiming_uniform_(layer1.weight, nonlinearity='leaky_relu')
            torch.nn.init.kaiming_uniform_(layer2.weight, nonlinearity='leaky_relu')

            self.cluster_layers1.append(nn.Sequential(layer1, bn1))
            self.cluster_layers2.append(nn.Sequential(layer2, bn2))

    def forward(self,x: Tensor, batch: Optional[Tensor],pos,
                     size: Optional[int] = None)-> Tensor:
        self.dim = -1 if isinstance(x, Tensor) and x.dim() == 1 else -2
        self.linear_model=self.linear_model.to(x.device)
        cluster_probs =nn.functional.softmax(self.linear_model(pos), dim=-1)
        _, cluster_indices = torch.max(cluster_probs, dim=-1)

        cluster_representations = []
        for i in range(self.c):
            mask = (cluster_indices == i).float().reshape(-1,1)  
            cluster_features = x * mask 
            cluster_out=scatter(cluster_features, batch, dim=self.dim, dim_size=size, reduce='mean')
            cluster_out=self.cluster_layers1[i](cluster_out)
            cluster_out=self.relu(cluster_out)
            cluster_out=F.dropout(cluster_out, p=self.droupout) 
            cluster_out=self.cluster_layers2[i](cluster_out)
            cluster_out=self.relu(cluster_out)
            cluster_out=F.dropout(cluster_out, p=self.droupout) 
            cluster_representations.append(cluster_out)

        out=torch.cat(cluster_representations,dim=1)

        return out,cluster_indices,