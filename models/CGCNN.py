from typing import Tuple, Union
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d, Linear
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor


class CGConv(MessagePassing):
    def __init__(self, inchannels: int, outchannels: int, 
                 dim: int = 0, aggr: str = 'add', batch_norm: bool = False,
                 bias: bool = True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.channels = (inchannels, outchannels)
        self.dim = dim
        self.batch_norm = batch_norm
        self.bias = bias
        
        channels = self.channels
        self.lin_skip = Linear(channels[0], channels[1], bias=bias)
        
        self.lin_f = Linear(2*channels[0] + dim, channels[1], bias=bias)
        self.lin_s = Linear(2*channels[0] + dim, channels[1], bias=bias)
        if batch_norm:
            self.bn = BatchNorm1d(channels[1])
        else:
            self.bn = None

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_f.reset_parameters()
        self.lin_s.reset_parameters()
        if self.bn is not None:
            self.bn.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = out if self.bn is None else self.bn(out)
        out = out + self.lin_skip(x[1])
        return out

    def message(self, x_i, x_j, edge_attr: OptTensor) -> Tensor:
        if edge_attr is None:
            z = torch.cat([x_i, x_j], dim=-1)
        else:
            z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.lin_f(z).sigmoid() * F.softplus(self.lin_s(z))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.channels}, dim={self.dim})'


class CGCNN(nn.Module):
    def __init__(self, edge_dim=14, out_dim=201*4, seed=123):
        super().__init__()
        self.seed = seed
        self.set_seed(self.seed)

        self.node_embedding = nn.Embedding(num_embeddings=118, embedding_dim=256)

        self.node_ln = nn.LayerNorm(256)
        
        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256)
        )
        self.edge_ln = nn.LayerNorm(256)
        
        
        self.fc_energies = nn.Sequential(
            nn.Linear(201, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256*3 + 128, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, out_dim)
        )
        self.leaky_relu = nn.LeakyReLU()
        self.init_weights()
        
        self.layers = nn.ModuleList([
            CGConv(inchannels = 256, outchannels = 256, dim=256, aggr='add', batch_norm=False, bias = False)
            for _ in range(3)
        ])        

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, data):
        '''
        Forward propagation
        
        Parameters
        ----------
        data: torch_geometric.data.Data
        The graph data object that contains the node feature (x), edge index (edge_index), edge feature (edge_attr), and energies
        '''
        x, edge_index, edge_attr, energies = data.x.long(), data.edge_index, data.edge_attr, data.energies
        batch = data.batch  

        x = self.node_embedding(x.squeeze(1))
        
        edge_attr = self.edge_embedding(edge_attr)
        
        for layer in self.layers:
            x = layer(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # Pooling operation: Mean, Max, and sum pooled for each graph, and concatenate the results
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        sum_pool = global_add_pool(x, batch)
        crys_fea = torch.cat((mean_pool, max_pool, sum_pool), dim=-1)

        energies = self.fc_energies(energies)
        
        crys_fea = torch.cat((crys_fea, energies), dim=-1)

        out = self.fc(crys_fea)

        num_graphs = batch.max().item() + 1
        
        return out.reshape(num_graphs, 4, 201)