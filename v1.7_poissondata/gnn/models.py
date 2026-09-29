import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, SAGEConv, global_mean_pool

class RadiationSensorGNN(torch.nn.Module):
    def __init__(self, in_channels=6, hidden_dim=512, out_channels=3):
        super(RadiationSensorGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.out = torch.nn.Linear(hidden_dim // 2, out_channels)

    def forward(self, x, edge_index, edge_weight, batch):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.relu(self.conv3(x, edge_index, edge_weight))
        
        x = global_mean_pool(x, batch)
        
        x = F.relu(self.fc1(x))
        return self.out(x)




class RadiationSensorGAT(torch.nn.Module):
    def __init__(self, in_channels=6, hidden_dim=128, out_channels=3, heads=4):
        super(RadiationSensorGAT, self).__init__()
        
       
        self.conv1 = GATv2Conv(in_channels, hidden_dim, heads=heads, edge_dim=1, concat=True)
        self.conv2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, edge_dim=1, concat=True)
        
        self.conv3 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, edge_dim=1, concat=False)
        
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.out = torch.nn.Linear(hidden_dim // 2, out_channels)

    def forward(self, x, edge_index, edge_weight, batch):
       
        if edge_weight.dim() == 1:
            edge_weight = edge_weight.unsqueeze(-1)

       
        x = F.relu(self.conv1(x, edge_index, edge_attr=edge_weight))
       
        
        x = F.relu(self.conv2(x, edge_index, edge_attr=edge_weight))
       
        
        x = F.relu(self.conv3(x, edge_index, edge_attr=edge_weight))
        
        x = global_mean_pool(x, batch)
        
        x = F.relu(self.fc1(x))
        return self.out(x)



class DynamicGNN(torch.nn.Module):
    def __init__(self, model_type='SAGE', num_layers=5, in_channels=6, hidden_dim=512, out_channels=3, heads=1, dropout=0.000159):
        super(DynamicGNN, self).__init__()
        self.model_type = model_type
        self.dropout = dropout
        self.num_layers = num_layers
        
        self.convs = torch.nn.ModuleList()
        current_dim = in_channels
        
        for i in range(num_layers - 1):
            if self.model_type == 'GAT':
                self.convs.append(GATv2Conv(current_dim, hidden_dim, heads=heads, edge_dim=1, concat=True))
                current_dim = hidden_dim * heads
            elif self.model_type == 'GCN':
                self.convs.append(GCNConv(current_dim, hidden_dim))
                current_dim = hidden_dim
            elif self.model_type == 'SAGE':
                self.convs.append(SAGEConv(current_dim, hidden_dim))
                current_dim = hidden_dim
                
       
        if self.model_type == 'GAT':
            self.convs.append(GATv2Conv(current_dim, hidden_dim, heads=1, edge_dim=1, concat=False))
        elif self.model_type == 'GCN':
            self.convs.append(GCNConv(current_dim, hidden_dim))
        elif self.model_type == 'SAGE':
            self.convs.append(SAGEConv(current_dim, hidden_dim))
            
        current_dim = hidden_dim
        
       
        self.fc1 = torch.nn.Linear(current_dim, current_dim // 2)
        self.out = torch.nn.Linear(current_dim // 2, out_channels)

    def forward(self, x, edge_index, edge_weight, batch):
        if self.model_type == 'GAT' and edge_weight.dim() == 1:
            edge_weight = edge_weight.unsqueeze(-1)

        for i, conv in enumerate(self.convs):
            if self.model_type == 'GAT':
                x = F.relu(conv(x, edge_index, edge_attr=edge_weight))
            elif self.model_type == 'GCN':
                x = F.relu(conv(x, edge_index, edge_weight))
            elif self.model_type == 'SAGE':
                x = F.relu(conv(x, edge_index))
            
       
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
                
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        return self.out(x)        