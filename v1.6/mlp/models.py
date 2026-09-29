import torch
import torch.nn as nn
import torch.nn.functional as F

class RadiationMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, out_dim=3, dropout=0.1):
        super(RadiationMLP, self).__init__()
        
        # A wide, stable architecture perfect for tabular data
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim) # Batch norm stabilizes tabular training
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        
        # Output: [source_x, source_y, log_I0]
        self.out = nn.Linear(hidden_dim // 2, out_dim)
        self.dropout = dropout

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return self.out(x)