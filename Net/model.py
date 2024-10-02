from torch_geometric.nn import global_mean_pool
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from .network import CustomGATLayer,CustomGCNLayer,CustomSageLayer
from .pooling import global_cluster_pool


class Model(nn.Module):
    def __init__(self, roi,hidden_dim,dropout_probab,num_layers,layer,cdim,ncluster):
        super(Model, self).__init__() 

        self.dropout = dropout_probab
        self.num_layers=num_layers
        self.convs = nn.ModuleList()    
        self.bn=nn.BatchNorm1d(hidden_dim)
        self.poolings=nn.ModuleList([global_cluster_pool(roi,hidden_dim,cdim,ncluster=ncluster,dropout=self.dropout) for _ in range(num_layers)])

        input_dim = 1
        for i in range(self.num_layers):
            if layer == 'GAT':
                self.convs.append(CustomGATLayer(input_dim, hidden_dim, 1,))
            elif layer == 'GCN':
                self.convs.append(CustomGCNLayer(input_dim, hidden_dim))
            elif layer == 'Sage':
                self.convs.append(CustomSageLayer(input_dim, hidden_dim))
            
            input_dim = hidden_dim 

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(cdim*num_layers*ncluster+4, hidden_dim),nn.ReLU(),nn.Dropout(0.5),
        nn.Linear(hidden_dim, 1))

    def forward(self,  x,edge_index,data):
        edge_attr, batch, pos =  data.edge_attr, data.batch ,data.pos
        out=x[:,0].reshape(-1,1)
        cov=x[:,1:].reshape(-1,4)

        layer_outs=[]
        for layers,poolings in zip(self.convs,self.poolings):
            out,edge_index,edge_attr,batch = layers(out, edge_index,edge_attr,batch)
            out = F.dropout(out, p=self.dropout, training=self.training) 
            mean_pool,cluster_indices = poolings(out, batch,pos=pos) 
            layer_outs.append(mean_pool)

        out = torch.cat(layer_outs, dim=1) 
        cov = global_mean_pool(cov,batch)
        out= torch.cat([out,cov],dim=1)
        out = self.post_mp(out) 

        return out.reshape(-1)

