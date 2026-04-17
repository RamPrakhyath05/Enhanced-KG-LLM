# models/rgcn.py
# Architecture unchanged — the improvement is using the output (see graph_utils.py)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class RGCN(nn.Module):
    def __init__(self, num_nodes, num_relations, dim=64):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, dim)
        self.conv1 = RGCNConv(dim, dim, num_relations)
        self.conv2 = RGCNConv(dim, dim, num_relations)

    def forward(self, edge_index, edge_type):
        x = self.embedding.weight
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_type)
        return x
