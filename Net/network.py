from torch_geometric.nn import GATConv,SAGEConv,GCNConv
import torch.nn as nn
import torch.nn.functional as F

class CustomGATLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim,edge_dim,):
        super(CustomGATLayer, self).__init__()
        self.conv = GATConv(input_dim, hidden_dim, edge_dim=edge_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index,edge_attr,batch):
        x = self.conv(x, edge_index,edge_attr)
        x = self.bn(x)
        x = self.relu(x)
        return x,edge_index,edge_attr,batch,
    
class CustomGCNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CustomGCNLayer, self).__init__()
        self.conv = GCNConv(input_dim, hidden_dim, normalize=False)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index,edge_attr,batch):
        x = self.conv(x, edge_index,edge_attr)
        x = self.bn(x)
        x = self.relu(x)
        return x,edge_index,edge_attr,batch
    
class CustomSageLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CustomSageLayer, self).__init__()
        self.conv = SAGEConv(input_dim, hidden_dim, )
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index,edge_attr,batch):
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = self.relu(x)
        return x,edge_index,edge_attr,batch