import math
import typing
from typing import Optional, Tuple, Union
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from typing import Tuple, Union
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import (
    Adj,
    NoneType,
    OptTensor,
    PairTensor,
    SparseTensor,
)
from torch_geometric.utils import softmax

if typing.TYPE_CHECKING:
    from typing import overload
else:
    from torch.jit import _overload_method as overload


class TransformerConv(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.bias = bias
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels, bias=bias)
        self.lin_query = Linear(in_channels[1], heads * out_channels,
                                bias=bias)
        self.lin_value = Linear(in_channels[0], heads * out_channels,
                                bias=bias)
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=bias)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=bias)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=bias)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    @overload
    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        return_attention_weights: NoneType = None,
    ) -> Tensor:
        pass

    @overload
    def forward(  # noqa: F811
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Tensor,
        edge_attr: OptTensor = None,
        return_attention_weights: bool = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        pass

    @overload
    def forward(  # noqa: F811
        self,
        x: Union[Tensor, PairTensor],
        edge_index: SparseTensor,
        edge_attr: OptTensor = None,
        return_attention_weights: bool = None,
    ) -> Tuple[Tensor, SparseTensor]:
        pass

    def forward(  # noqa: F811
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        return_attention_weights: Optional[bool] = None,
    ) -> Union[
            Tensor,
            Tuple[Tensor, Tuple[Tensor, Tensor]],
            Tuple[Tensor, SparseTensor],
    ]:
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor or (torch.Tensor, torch.Tensor)): The input node
                features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            x = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor,
        #                  edge_attr: OptTensor)
        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out = out + x_r

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                      self.out_channels)
            key_j = key_j + edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        if edge_attr is not None:
            out = out + edge_attr

        out = out * alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')

class CGT(nn.Module):
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
            TransformerConv(
                in_channels = 256,
                out_channels = 256,
                heads = 4,
                concat = False,
                edge_dim = 256, 
                bias = False,
                root_weight = True
            )
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