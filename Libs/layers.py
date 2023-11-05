########### - Imports - ###########
#################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Parameter
from torch.nn.init import xavier_uniform_, zeros_
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.utils import add_self_loops, remove_self_loops, degree, softmax
from typing import Type


########### - End - ###########
#################################################################################


########### - Custom GNN Layers - ###########
#################################################################################
class EINv1(MessagePassing):
    """
    A Edge featured attention based Graph Neural Network Layer for Graph Classification / Regression Tasks: V1
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            heads=1,
            negative_slope=0.2,
            dropout=0.0,
            edge_dim=None,
            train_eps=False,
            eps=0.0,
            bias=True,
            share_weights=False,
            **kwargs,
    ):
        super().__init__(node_dim=0, aggr='add', **kwargs)  # defines the aggregation method: `aggr='add'`

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.share_weights = share_weights
        self.edge_dim = edge_dim
        self.initial_eps = eps

        # Linear Transformation
        self.lin_l = Linear(in_channels, heads * out_channels, bias=bias)

        if share_weights:
            self.lin_r = self.lin_l  # use same matrix
        else:
            self.lin_r = Linear(in_channels, heads * out_channels, bias=bias)

        # For attention calculation
        self.att = Parameter(torch.Tensor(1, heads, out_channels))

        # For influence mechanism
        self.inf = Linear(edge_dim, out_channels)

        # Tunable parameter for adding self node features...
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None  # alpha weights

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        self.inf.reset_parameters()
        self.eps.data.fill_(self.initial_eps)
        xavier_uniform_(self.att)
        zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr=None, return_attention_weights=None):
        ## N - no_of_nodes, NH - no_of heads,  H_in - input_channels, H_out - out_channels

        H, C = self.heads, self.out_channels

        x_l = None  # for source nodes
        x_r = None  # for target nodes

        x_l = self.lin_l(x).view(-1, H, C)  # (N, H_in) -> (N, NH, H_Out)
        if self.share_weights:
            x_r = x_l
        else:
            x_r = self.lin_r(x).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        # Check the edge features shape: test_case
        # if edge_attr is not None:
        #     print(f'edge_features shape: {edge_attr.shape}')
        # else:
        #     print('No edge features!')

        # Start propagating info...: construct message -> aggregate message -> update/obtain new representations
        out = self.propagate(edge_index, x=(x_l, x_r), edge_attr=edge_attr, size=None)  # (N, H_out)
        # out += x_r.mean(dim=1) # add the self features

        alpha = self._alpha  # (#edges, 1)
        assert alpha is not None, 'Alpha weights can not be None value!'

        if self.bias is not None:
            out = out + self.bias

        # Returning attention weights with computed hidden features
        if isinstance(return_attention_weights, bool):
            return out, alpha.mean(dim=1, keepdims=True)
        else:
            return out  # (N, H_out)

    def message(self, x_j, x_i, index, size_i, edge_attr):
        # x_j has shape [#edges, NH, H_out]
        # x_i has shape [#edges, NH, H_out]
        # index: target node indexes, where data flows 'source_to_target': this is for computing softmax
        # size: size_i, size_j mean num_nodes in the graph

        x = x_i + x_j  # adding(element-wise) source and target node features together to calculate attention
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)  # (#edges, NH)
        alpha = softmax(alpha, index,
                        num_nodes=size_i)  # spares softmax: groups node's attention and then node-wise softmax
        self._alpha = alpha.mean(dim=1, keepdims=True)  # (#edges, 1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)  # randomly dropping attention during training
        node_out = (x_j * alpha.unsqueeze(-1)).mean(dim=1)

        if self.inf is not None and edge_attr is not None:
            if self.edge_dim != edge_attr.size(-1):
                raise ValueError("Node and edge feature dimensionality do not "
                                 "match. Consider setting the 'edge_dim' ""attribute")
            edge_attr = self.inf(self._alpha * edge_attr)  # transformed edge features via influence mechanism
            return node_out + edge_attr  # (#edges, H_out)
        return node_out  # (#edges, H_out)

    def update(self, aggr_out, x):
        aggr_out += (1 + self.eps) * x[1].mean(dim=1)  # add the self features with a weighting factor
        return aggr_out  # (N, H_out)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class EINv2(MessagePassing):
    """
    A Edge featured attention based Graph Neural Network Layer(+MLP) for Graph Classification / Regression Tasks: V2

    2-layer MLP Block is used to learn more features following the model arch: GIN
    """

    def __init__(
            self,
            nn,
            in_channels,
            out_channels,
            heads=1,
            negative_slope=0.2,
            dropout=0.0,
            edge_dim=None,
            train_eps=False,
            eps=0.0,
            bias=True,
            share_weights=False,
            **kwargs,
    ):
        super().__init__(node_dim=0, aggr='add', **kwargs)  # defines the aggregation method: `aggr='add'`
        self.nn = nn
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.share_weights = share_weights
        self.edge_dim = edge_dim
        self.initial_eps = eps

        # Linear Transformation
        self.lin_l = Linear(in_channels, heads * out_channels, bias=bias)

        if share_weights:
            self.lin_r = self.lin_l  # use same matrix
        else:
            self.lin_r = Linear(in_channels, heads * out_channels, bias=bias)

        # For attention calculation
        self.att = Parameter(torch.Tensor(1, heads, out_channels))

        # For influence mechanism
        self.inf = Linear(edge_dim, out_channels)

        # Tunable parameter for adding self node features...
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None  # alpha weights

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.nn)
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        self.inf.reset_parameters()
        self.eps.data.fill_(self.initial_eps)
        xavier_uniform_(self.att)
        zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr=None, return_attention_weights=None):
        ## N - no_of_nodes, NH - no_of heads,  H_in - input_channels, H_out - out_channels

        H, C = self.heads, self.out_channels

        x_l = None  # for source nodes
        x_r = None  # for target nodes

        x_l = self.lin_l(x).view(-1, H, C)  # (N, H_in) -> (N, NH, H_Out)
        if self.share_weights:
            x_r = x_l
        else:
            x_r = self.lin_r(x).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        # Check the edge features shape: test_case
        # if edge_attr is not None:
        #     print(f'edge_features shape: {edge_attr.shape}')
        # else:
        #     print('No edge features!')

        # Start propagating info...: construct message -> aggregate message -> update/obtain new representations
        out = self.propagate(edge_index, x=(x_l, x_r), edge_attr=edge_attr, size=None)  # (N, H_out)
        # out += x_r.mean(dim=1) # add the self features

        alpha = self._alpha  # (#edges, 1)
        assert alpha is not None, 'Alpha weights can not be None value!'

        if self.bias is not None:
            out = out + self.bias

        # Returning attention weights with computed hidden features
        if isinstance(return_attention_weights, bool):
            return out, alpha.mean(dim=1, keepdims=True)
        else:
            return self.nn(out)  # (N, H_out)

    def message(self, x_j, x_i, index, size_i, edge_attr):
        # x_j has shape [#edges, NH, H_out]
        # x_i has shape [#edges, NH, H_out]
        # index: target node indexes, where data flows 'source_to_target': this is for computing softmax
        # size: size_i, size_j mean num_nodes in the graph

        x = x_i + x_j  # adding(element-wise) source and target node features together to calculate attention
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)  # (#edges, NH)
        alpha = softmax(alpha, index,
                        num_nodes=size_i)  # spares softmax: groups node's attention and then node-wise softmax
        self._alpha = alpha.mean(dim=1, keepdims=True)  # (#edges, 1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)  # randomly dropping attention during training
        node_out = (x_j * alpha.unsqueeze(-1)).mean(dim=1)

        if self.inf is not None and edge_attr is not None:
            if self.edge_dim != edge_attr.size(-1):
                raise ValueError("Node and edge feature dimensionality do not "
                                 "match. Consider setting the 'edge_dim' ""attribute")
            edge_attr = self.inf(self._alpha * edge_attr)  # transformed edge features via influence mechanism
            return node_out + edge_attr  # (#edges, H_out)
        return node_out  # (#edges, H_out)

    def update(self, aggr_out, x):
        aggr_out += (1 + self.eps) * x[1].mean(dim=1)  # add the self features with a weighting factor
        return aggr_out  # (N, H_out)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class EINv3(MessagePassing):
    """
    A Edge featured attention based Graph Neural Network Layer for Graph Classification / Regression Tasks: V3
    
    Notes:
        Fully Multi-head attention is implemented in this version, compared to previous versions where concatenation is ommited and mean of values are used.
        In this case, mean value of attention is used only with influence mechanism!
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            heads=1,
            negative_slope=0.2,
            dropout=0.0,
            edge_dim=None,
            train_eps=False,
            eps=0.0,
            bias=True,
            share_weights=False,
            concat=True,
            **kwargs,
    ):
        super().__init__(node_dim=0, aggr='add', **kwargs)  # defines the aggregation method: `aggr='add'`

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.share_weights = share_weights
        self.edge_dim = edge_dim
        self.initial_eps = eps
        self.concat = concat

        # Linear Transformation
        self.lin_l = Linear(in_channels, heads * out_channels, bias=bias)

        if share_weights:
            self.lin_r = self.lin_l  # use same matrix
        else:
            self.lin_r = Linear(in_channels, heads * out_channels, bias=bias)

        # For attention calculation
        self.att = Parameter(torch.Tensor(1, heads, out_channels))

        # For influence mechanism
        self.inf = Linear(edge_dim, out_channels)

        # Tunable parameter for adding self node features...
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None  # alpha weights

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        self.inf.reset_parameters()
        self.eps.data.fill_(self.initial_eps)
        xavier_uniform_(self.att)
        zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr=None, return_attention_weights=None):
        ## N - no_of_nodes, NH - no_of heads,  H_in - input_channels, H_out - out_channels

        H, C = self.heads, self.out_channels

        x_l = None  # for source nodes
        x_r = None  # for target nodes

        x_l = self.lin_l(x).view(-1, H, C)  # (N, H_in) -> (N, NH, H_Out)
        if self.share_weights:
            x_r = x_l
        else:
            x_r = self.lin_r(x).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        # Check the edge features shape: test_case
        # if edge_attr is not None:
        #     print(f'edge_features shape: {edge_attr.shape}')
        # else:
        #     print('No edge features!')

        # Start propagating info...: construct message -> aggregate message -> update/obtain new representations
        out = self.propagate(edge_index, x=(x_l, x_r), edge_attr=edge_attr, size=None)  # (N, H_out)
        # out += x_r.mean(dim=1) # add the self features
        alpha = self._alpha  # (#edges, NH, H_out)
        assert alpha is not None, 'Alpha weights can not be None value!'

        if self.bias is not None:
            out = out + self.bias

        if self.concat == True: # (N,  NH * H_out), for alpha: (N, NH, H_out)
            if isinstance(return_attention_weights, bool):
                return out.view(-1, self.heads * self.out_channels), alpha
            return out.view(-1, self.heads * self.out_channels)
        
        # Taking the mean of heads  -> (N, H_out)
        if isinstance(return_attention_weights, bool):
            return out.mean(dim=1), alpha.mean(dim=1)

        return out.mean(dim=1)


    def message(self, x_j, x_i, index, size_i, edge_attr):
        # x_j has shape [#edges, NH, H_out]
        # x_i has shape [#edges, NH, H_out]
        # index: target node indexes, where data flows 'source_to_target': this is for computing softmax
        # size: size_i, size_j mean num_nodes in the graph

        x = x_i + x_j  # adding(element-wise) source and target node features together to calculate attention
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1) # (#edges, NH)
        alpha = softmax(alpha, index, num_nodes=size_i)  # spares softmax: groups node's attention and then node-wise softmax
        self._alpha = alpha  # (#edges, NH)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)  # randomly dropping attention during training
        node_out = x_j * alpha.unsqueeze(dim=-1)

        if self.inf is not None and edge_attr is not None:
            if self.edge_dim != edge_attr.size(-1):
                raise ValueError("Node and edge feature dimensionality do not "
                                 "match. Consider setting the 'edge_dim' ""attribute")
            edge_attr = self.inf(self._alpha.mean(dim=-1, keepdim=True) * edge_attr)  # transformed edge features via influence mechanism
            return node_out + edge_attr.unsqueeze(1)  # (#edges, H_out)
        return node_out  # (#edges, H_out)


    def update(self, aggr_out, x):
        aggr_out += (1 + self.eps) * x[1]  # add the self features with a weighting factor
        return aggr_out  # (N, H_out)


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class EINv4(MessagePassing):
    """
    A Edge featured attention based Graph Neural Network Layer for Graph Classification / Regression Tasks: V4

    Note: 
        - Centrality encoding is added: in-degree, out-degree
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            heads=1,
            negative_slope=0.2,
            dropout=0.0,
            edge_dim=None,
            train_eps=False,
            eps=0.0,
            bias=True,
            share_weights=False,
            **kwargs,
    ):
        super().__init__(node_dim=0, aggr='add', **kwargs)  # defines the aggregation method: `aggr='add'`

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.share_weights = share_weights
        self.edge_dim = edge_dim
        self.initial_eps = eps

        # Linear Transformation
        self.lin_l = Linear(in_channels, heads * out_channels, bias=bias)

        if share_weights:
            self.lin_r = self.lin_l  # use same matrix
        else:
            self.lin_r = Linear(in_channels, heads * out_channels, bias=bias)

        # For attention calculation
        self.att = Parameter(torch.Tensor(1, heads, out_channels))

        # For influence mechanism
        self.inf = Linear(edge_dim, out_channels)

        # Tunable parameter for adding self node features...
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        
        # In-degree and Out-degree encoders
        self.in_degree_encoder = nn.Embedding(out_channels, out_channels, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(out_channels, out_channels, padding_idx=0)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None  # alpha weights

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        self.inf.reset_parameters()
        self.in_degree_encoder.reset_parameters()
        self.out_degree_encoder.reset_parameters()
        self.eps.data.fill_(self.initial_eps)
        xavier_uniform_(self.att)
        zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr=None, return_attention_weights=None):
        ## N - no_of_nodes, NH - no_of heads,  H_in - input_channels, H_out - out_channels

        H, C = self.heads, self.out_channels

        x_l = None  # for source nodes
        x_r = None  # for target nodes

        x_l = self.lin_l(x).view(-1, H, C)  # (N, H_in) -> (N, NH, H_Out)
        if self.share_weights:
            x_r = x_l
        else:
            x_r = self.lin_r(x).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        # Start propagating info...: construct message -> aggregate message -> update/obtain new representations
        out = self.propagate(edge_index, x=(x_l, x_r), edge_attr=edge_attr, size=None)  # (N, H_out)
        # out += x_r.mean(dim=1) # add the self features

        alpha = self._alpha  # (#edges, 1)
        assert alpha is not None, 'Alpha weights can not be None value!'

        if self.bias is not None:
            out = out + self.bias

        # Add in-degree and out-degree informations
        in_degree = degree(edge_index[1]).to(torch.long)
        out_degree = degree(edge_index[0]).to(torch.long)
        out = (
            out 
            + self.in_degree_encoder(in_degree) 
            + self.out_degree_encoder(out_degree)
        )

        # Returning attention weights with computed hidden features
        if isinstance(return_attention_weights, bool):
            return out, alpha.mean(dim=1, keepdims=True)
        else:
            return out  # (N, H_out)

    def message(self, x_j, x_i, index, size_i, edge_attr):
        # x_j has shape [#edges, NH, H_out]
        # x_i has shape [#edges, NH, H_out]
        # index: target node indexes, where data flows 'source_to_target': this is for computing softmax
        # size: size_i, size_j mean num_nodes in the graph

        x = x_i + x_j  # adding(element-wise) source and target node features together to calculate attention
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)  # (#edges, NH)
        alpha = softmax(alpha, index,
                        num_nodes=size_i)  # spares softmax: groups node's attention and then node-wise softmax
        self._alpha = alpha.mean(dim=1, keepdims=True)  # (#edges, 1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)  # randomly dropping attention during training
        node_out = (x_j * alpha.unsqueeze(-1)).mean(dim=1)

        if self.inf is not None and edge_attr is not None:
            if self.edge_dim != edge_attr.size(-1):
                raise ValueError("Node and edge feature dimensionality do not "
                                 "match. Consider setting the 'edge_dim' ""attribute")
            edge_attr = self.inf(self._alpha * edge_attr)  # transformed edge features via influence mechanism
            return node_out + edge_attr  # (#edges, H_out)
        return node_out  # (#edges, H_out)

    def update(self, aggr_out, x):
        aggr_out += (1 + self.eps) * x[1].mean(dim=1)  # add the self features with a weighting factor
        return aggr_out  # (N, H_out)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class EINv5(MessagePassing):
    """
    A Edge featured attention based Graph Neural Network Layer(+MLP) for Graph Classification / Regression Tasks: V2

    2-layer MLP Block is used to learn more features following the model arch: GIN

    Note: 
        - Centrality encoding is added: in-degree, out-degree
    """

    def __init__(
            self,
            nn,
            in_channels,
            out_channels,
            heads=1,
            negative_slope=0.2,
            dropout=0.0,
            edge_dim=None,
            train_eps=False,
            eps=0.0,
            bias=True,
            share_weights=False,
            **kwargs,
    ):
        super().__init__(node_dim=0, aggr='add', **kwargs)  # defines the aggregation method: `aggr='add'`
        self.nn = nn
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.share_weights = share_weights
        self.edge_dim = edge_dim
        self.initial_eps = eps

        # Linear Transformation
        self.lin_l = Linear(in_channels, heads * out_channels, bias=bias)

        if share_weights:
            self.lin_r = self.lin_l  # use same matrix
        else:
            self.lin_r = Linear(in_channels, heads * out_channels, bias=bias)

        # For attention calculation
        self.att = Parameter(torch.Tensor(1, heads, out_channels))

        # For influence mechanism
        self.inf = Linear(edge_dim, out_channels)

        # Tunable parameter for adding self node features...
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        # In-degree and Out-degree encoders
        self.in_degree_encoder = torch.nn.Embedding(out_channels, out_channels, padding_idx=0)
        self.out_degree_encoder = torch.nn.Embedding(out_channels, out_channels, padding_idx=0)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None  # alpha weights

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.nn)
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        self.inf.reset_parameters()
        self.in_degree_encoder.reset_parameters()
        self.out_degree_encoder.reset_parameters()
        self.eps.data.fill_(self.initial_eps)
        xavier_uniform_(self.att)
        zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr=None, return_attention_weights=None):
        ## N - no_of_nodes, NH - no_of heads,  H_in - input_channels, H_out - out_channels

        H, C = self.heads, self.out_channels

        x_l = None  # for source nodes
        x_r = None  # for target nodes

        x_l = self.lin_l(x).view(-1, H, C)  # (N, H_in) -> (N, NH, H_Out)
        if self.share_weights:
            x_r = x_l
        else:
            x_r = self.lin_r(x).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        # Check the edge features shape: test_case
        # if edge_attr is not None:
        #     print(f'edge_features shape: {edge_attr.shape}')
        # else:
        #     print('No edge features!')

        # Start propagating info...: construct message -> aggregate message -> update/obtain new representations
        out = self.propagate(edge_index, x=(x_l, x_r), edge_attr=edge_attr, size=None)  # (N, H_out)
        # out += x_r.mean(dim=1) # add the self features

        alpha = self._alpha  # (#edges, 1)
        assert alpha is not None, 'Alpha weights can not be None value!'

        if self.bias is not None:
            out = out + self.bias
        
         # Add in-degree and out-degree informations
        in_degree = degree(edge_index[1]).to(torch.long)
        out_degree = degree(edge_index[0]).to(torch.long)
        out = (
            out 
            + self.in_degree_encoder(in_degree) 
            + self.out_degree_encoder(out_degree)
        )

        # Returning attention weights with computed hidden features
        if isinstance(return_attention_weights, bool):
            return self.nn(out), alpha.mean(dim=1, keepdims=True)
        else:
            return self.nn(out)  # (N, H_out)

    def message(self, x_j, x_i, index, size_i, edge_attr):
        # x_j has shape [#edges, NH, H_out]
        # x_i has shape [#edges, NH, H_out]
        # index: target node indexes, where data flows 'source_to_target': this is for computing softmax
        # size: size_i, size_j mean num_nodes in the graph

        x = x_i + x_j  # adding(element-wise) source and target node features together to calculate attention
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)  # (#edges, NH)
        alpha = softmax(alpha, index,
                        num_nodes=size_i)  # spares softmax: groups node's attention and then node-wise softmax
        self._alpha = alpha.mean(dim=1, keepdims=True)  # (#edges, 1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)  # randomly dropping attention during training
        node_out = (x_j * alpha.unsqueeze(-1)).mean(dim=1)

        if self.inf is not None and edge_attr is not None:
            if self.edge_dim != edge_attr.size(-1):
                raise ValueError("Node and edge feature dimensionality do not "
                                 "match. Consider setting the 'edge_dim' ""attribute")
            edge_attr = self.inf(self._alpha * edge_attr)  # transformed edge features via influence mechanism
            return node_out + edge_attr  # (#edges, H_out)
        return node_out  # (#edges, H_out)

    def update(self, aggr_out, x):
        aggr_out += (1 + self.eps) * x[1].mean(dim=1)  # add the self features with a weighting factor
        return aggr_out  # (N, H_out)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')

########### - End - ###########
#################################################################################
