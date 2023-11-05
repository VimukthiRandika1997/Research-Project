################### - Imports - #############################################################
import torch
from torch.nn import Linear, Parameter, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import MessagePassing, GCNConv, GATv2Conv, GINConv, GINEConv, global_mean_pool
from torch_geometric.utils import add_self_loops, remove_self_loops, degree, softmax


# Custom imports
import sys
sys.path.append('./')
from Libs.layers import *
from Libs.common import *

##################################
############################################################################################


################### - Models - #############################################################

#### - EINv1 - ###
class EINModel(torch.nn.Module):
    def __init__(self, input_dim, dim_h, final_dim, num_heads, edge_dim, **kwargs):
        super().__init__()
        torch.manual_seed(42)

        # Layers
        self.conv1 = EINv1(input_dim, dim_h, edge_dim=edge_dim,
                           heads=num_heads, **kwargs)

        self.conv2 = EINv1(dim_h, dim_h, edge_dim=edge_dim,
                           heads=num_heads, **kwargs)

        self.conv3 = EINv1(dim_h, dim_h, edge_dim=edge_dim,
                           heads=num_heads, **kwargs)

        self.conv4 = EINv1(dim_h, dim_h, edge_dim=edge_dim,
                           heads=num_heads, **kwargs)
                           
        self.conv5 = EINv1(dim_h, dim_h, edge_dim=edge_dim,
                           heads=num_heads, **kwargs)

        # Linear layer
        self.lin1 = Linear(dim_h * 5, dim_h * 5)

        # Classification head
        # final_dim: for classification or regression task
        self.lin2 = Linear(dim_h * 5, final_dim)

    def forward(self, x, edge_index, edge_attr, batch, type='binary'):
        # Embedding
        h1 = self.conv1(x, edge_index, edge_attr)
        h1 = h1.relu()
        h2 = self.conv2(h1, edge_index, edge_attr)
        h2 = h2.relu()
        h3 = self.conv3(h2, edge_index, edge_attr)
        h3 = h3.relu()
        h4 = self.conv4(h3, edge_index, edge_attr)
        h4 = h4.relu()
        h5 = self.conv5(h4, edge_index, edge_attr)
        h5 = h5.relu()

        # Graph-level readout
        h1 = global_mean_pool(h1, batch)
        h2 = global_mean_pool(h2, batch)
        h3 = global_mean_pool(h3, batch)
        h4 = global_mean_pool(h4, batch)
        h5 = global_mean_pool(h5, batch)

        h = torch.cat((h1, h2, h3, h4, h5), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        
        return h.flatten()
        # return F.log_softmax(h, dim=1)
###############


#### - EINv2 - ###
class EINModel_v2(torch.nn.Module):
    def __init__(self, input_dim, dim_h, final_dim, num_heads, edge_dim, **kwargs):
        super().__init__()
        torch.manual_seed(42)

        # Layers
        self.conv1 = EINv2(MLPBlockNorm(dim_h, dim_h), 
                           input_dim, 
                           dim_h, 
                           edge_dim=edge_dim,
                           heads=num_heads, 
                           **kwargs)
        
        self.conv2 = EINv2(MLPBlockNorm(dim_h, dim_h),
                            dim_h, 
                            dim_h, 
                            edge_dim=edge_dim,
                           heads=num_heads, 
                           **kwargs)
        
        self.conv3 = EINv2(MLPBlockNorm(dim_h, dim_h), 
                           dim_h, 
                           dim_h, 
                           edge_dim=edge_dim,
                           heads=num_heads, 
                           **kwargs)
        self.conv4 = EINv2(MLPBlockNorm(dim_h, dim_h), 
                           dim_h, 
                           dim_h, 
                           edge_dim=edge_dim,
                           heads=num_heads, 
                           **kwargs)
        self.conv5 = EINv2(MLPBlockNorm(dim_h, dim_h), 
                           dim_h, 
                           dim_h, 
                           edge_dim=edge_dim,
                           heads=num_heads, 
                           **kwargs)

        # Linear layer
        self.lin1 = Linear(dim_h * 5, dim_h * 5)

        # Classification head
        # final_dim: for classification or regression task
        self.lin2 = Linear(dim_h * 5, final_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        # Embedding
        h1 = self.conv1(x, edge_index, edge_attr)
        h1 = h1.relu()
        h2 = self.conv2(h1, edge_index, edge_attr)
        h2 = h2.relu()
        h3 = self.conv3(h2, edge_index, edge_attr)
        h3 = h3.relu()
        h4 = self.conv4(h3, edge_index, edge_attr)
        h4 = h4.relu()
        h5 = self.conv5(h4, edge_index, edge_attr)
        h5 = h5.relu()

        # Graph-level readout
        h1 = global_mean_pool(h1, batch)
        h2 = global_mean_pool(h2, batch)
        h3 = global_mean_pool(h3, batch)
        h4 = global_mean_pool(h4, batch)
        h5 = global_mean_pool(h5, batch)

        h = torch.cat((h1, h2, h3, h4, h5), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)

        # return F.log_softmax(h, dim=1)
        return h.flatten()
###############


#### - EINv3 - ###
class EINModel_v3(torch.nn.Module):
    def __init__(self, input_dim, dim_h, final_dim, num_heads, edge_dim, **kwargs):
        super().__init__()
        torch.manual_seed(42)

        # Layers
        self.conv1 = EINv3(input_dim, 
                           dim_h,
                           edge_dim=edge_dim, 
                           heads=num_heads, 
                           **kwargs)
        
        self.conv2 = EINv3(dim_h * num_heads, 
                               dim_h,
                               edge_dim=edge_dim, 
                               heads=num_heads, 
                               **kwargs)
        
        self.conv3 = EINv3(dim_h * num_heads, 
                               dim_h,
                               edge_dim=edge_dim, 
                               heads=num_heads, 
                               **kwargs)
        
        self.conv4 = EINv3(dim_h * num_heads, 
                               dim_h,
                               edge_dim=edge_dim, 
                               heads=num_heads, 
                               **kwargs)
        
        self.conv5 = EINv3(dim_h * num_heads, 
                               dim_h, 
                               edge_dim=edge_dim, 
                               heads=num_heads, 
                               concat=False, 
                               **kwargs)

        # Linear layer
        self.lin1 = Linear(dim_h * 5, dim_h * 5)

        # Classification head
        self.lin2 = Linear(dim_h * 5, final_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        # Embedding
        h1 = self.conv1(x, edge_index, edge_attr)
        h1 = h1.relu()
        h2 = self.conv2(h1, edge_index, edge_attr)
        h2 = h2.relu()
        h3 = self.conv3(h2, edge_index, edge_attr)
        h3 = h3.relu()
        h4 = self.conv4(h3, edge_index, edge_attr)
        h4 = h4.relu()
        h5 = self.conv5(h4, edge_index, edge_attr)
        h5 = h5.relu()

        C = h5.shape[-1]  # dim_h
        H = h4.shape[-1] // C  # num_heads

        # Graph-level readout
        h1 = global_mean_pool(h1.view(-1, H, C).mean(dim=1), batch)
        h2 = global_mean_pool(h2.view(-1, H, C).mean(dim=1), batch)
        h3 = global_mean_pool(h3.view(-1, H, C).mean(dim=1), batch)
        h4 = global_mean_pool(h4.view(-1, H, C).mean(dim=1), batch)
        h5 = global_mean_pool(h5, batch)

        h = torch.cat((h1, h2, h3, h4, h5), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)

        # return F.log_softmax(h, dim=1)
        return h.flatten()

###############

#### - EINv4 - ###
class EINModel_v4(torch.nn.Module):
    def __init__(self, input_dim, dim_h, final_dim, num_heads, edge_dim, **kwargs):
        super().__init__()
        torch.manual_seed(42)

        # Layers
        self.conv1 = EINv4(input_dim, dim_h, edge_dim=edge_dim,
                           heads=num_heads, **kwargs)
        self.conv2 = EINv4(dim_h, dim_h, edge_dim=edge_dim,
                           heads=num_heads, **kwargs)
        self.conv3 = EINv4(dim_h, dim_h, edge_dim=edge_dim,
                           heads=num_heads, **kwargs)
        self.conv4 = EINv4(dim_h, dim_h, edge_dim=edge_dim,
                           heads=num_heads, **kwargs)
        self.conv5 = EINv4(dim_h, dim_h, edge_dim=edge_dim,
                           heads=num_heads, **kwargs)

        # Linear layer
        self.lin1 = Linear(dim_h * 5, dim_h * 5)

        # Classification head
        # final_dim: for classification or regression task
        self.lin2 = Linear(dim_h * 5, final_dim)

    def forward(self, x, edge_index, edge_attr, batch, type='binary'):
        # Embedding
        h1 = self.conv1(x, edge_index, edge_attr)
        h1 = h1.relu()
        h2 = self.conv2(h1, edge_index, edge_attr)
        h2 = h2.relu()
        h3 = self.conv3(h2, edge_index, edge_attr)
        h3 = h3.relu()
        h4 = self.conv4(h3, edge_index, edge_attr)
        h4 = h4.relu()
        h5 = self.conv5(h4, edge_index, edge_attr)
        h5 = h5.relu()

        # Graph-level readout
        h1 = global_mean_pool(h1, batch)
        h2 = global_mean_pool(h2, batch)
        h3 = global_mean_pool(h3, batch)
        h4 = global_mean_pool(h4, batch)
        h5 = global_mean_pool(h5, batch)

        h = torch.cat((h1, h2, h3, h4, h5), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        
        return h.flatten()
        # return F.log_softmax(h, dim=1)
###############


#### - EINv5 - ###
class EINModel_v5(torch.nn.Module):
    def __init__(self, input_dim, dim_h, final_dim, num_heads, edge_dim, **kwargs):
        super().__init__()
        torch.manual_seed(42)

        # Layers
        self.conv1 = EINv5(MLPBlockNorm(dim_h, dim_h), 
                           input_dim, 
                           dim_h, 
                           edge_dim=edge_dim,
                           heads=num_heads, 
                           **kwargs)
        
        self.conv2 = EINv5(MLPBlockNorm(dim_h, dim_h),
                            dim_h, 
                            dim_h, 
                            edge_dim=edge_dim,
                           heads=num_heads, 
                           **kwargs)
        
        self.conv3 = EINv5(MLPBlockNorm(dim_h, dim_h), 
                           dim_h, 
                           dim_h, 
                           edge_dim=edge_dim,
                           heads=num_heads, 
                           **kwargs)
        self.conv4 = EINv5(MLPBlockNorm(dim_h, dim_h), 
                           dim_h, 
                           dim_h, 
                           edge_dim=edge_dim,
                           heads=num_heads, 
                           **kwargs)
        self.conv5 = EINv5(MLPBlockNorm(dim_h, dim_h), 
                           dim_h, 
                           dim_h, 
                           edge_dim=edge_dim,
                           heads=num_heads, 
                           **kwargs)

        # Linear layer
        self.lin1 = Linear(dim_h * 5, dim_h * 5)

        # Classification head
        # final_dim: for classification or regression task
        self.lin2 = Linear(dim_h * 5, final_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        # Embedding
        h1 = self.conv1(x, edge_index, edge_attr)
        h1 = h1.relu()
        h2 = self.conv2(h1, edge_index, edge_attr)
        h2 = h2.relu()
        h3 = self.conv3(h2, edge_index, edge_attr)
        h3 = h3.relu()
        h4 = self.conv4(h3, edge_index, edge_attr)
        h4 = h4.relu()
        h5 = self.conv5(h4, edge_index, edge_attr)
        h5 = h5.relu()

        # Graph-level readout
        h1 = global_mean_pool(h1, batch)
        h2 = global_mean_pool(h2, batch)
        h3 = global_mean_pool(h3, batch)
        h4 = global_mean_pool(h4, batch)
        h5 = global_mean_pool(h5, batch)

        h = torch.cat((h1, h2, h3, h4, h5), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)

        # return F.log_softmax(h, dim=1)
        return h.flatten()
###############


#### - GATv2 - ###
class GATModel(torch.nn.Module):
    def __init__(self, input_dim, dim_h, final_dim, num_heads, edge_dim, **kwargs):
        super().__init__()
        torch.manual_seed(42)

        # Layers
        self.conv1 = GATv2Conv(input_dim, dim_h,
                               edge_dim=edge_dim, heads=num_heads, **kwargs)
        self.conv2 = GATv2Conv(dim_h * num_heads, dim_h,
                               edge_dim=edge_dim, heads=num_heads, **kwargs)
        self.conv3 = GATv2Conv(dim_h * num_heads, dim_h,
                               edge_dim=edge_dim, heads=num_heads, **kwargs)
        self.conv4 = GATv2Conv(dim_h * num_heads, dim_h,
                               edge_dim=edge_dim, heads=num_heads, **kwargs)
        self.conv5 = GATv2Conv(
            dim_h * num_heads, dim_h, edge_dim=edge_dim, heads=num_heads, concat=False, **kwargs)

        # Linear layer
        self.lin1 = Linear(dim_h * 5, dim_h * 5)

        # Classification head
        self.lin2 = Linear(dim_h * 5, final_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        # Embedding
        h1 = self.conv1(x, edge_index, edge_attr)
        h1 = h1.relu()
        h2 = self.conv2(h1, edge_index, edge_attr)
        h2 = h2.relu()
        h3 = self.conv3(h2, edge_index, edge_attr)
        h3 = h3.relu()
        h4 = self.conv4(h3, edge_index, edge_attr)
        h4 = h4.relu()
        h5 = self.conv5(h4, edge_index, edge_attr)
        h5 = h5.relu()

        C = h5.shape[-1]  # dim_h
        H = h4.shape[-1] // C  # num_heads

        # Graph-level readout
        h1 = global_mean_pool(h1.view(-1, H, C).mean(dim=1), batch)
        h2 = global_mean_pool(h2.view(-1, H, C).mean(dim=1), batch)
        h3 = global_mean_pool(h3.view(-1, H, C).mean(dim=1), batch)
        h4 = global_mean_pool(h4.view(-1, H, C).mean(dim=1), batch)
        h5 = global_mean_pool(h5, batch)

        h = torch.cat((h1, h2, h3, h4, h5), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)

        # return F.log_softmax(h, dim=1)
        return h.flatten()

###############


#### - GCN - ###
class GCNModel(torch.nn.Module):
    def __init__(self, input_dim, dim_h, final_dim, **kwargs):
        super().__init__()
        torch.manual_seed(42)

        # Layers
        self.conv1 = GCNConv(input_dim, dim_h, **kwargs)
        self.conv2 = GCNConv(dim_h, dim_h, **kwargs)
        self.conv3 = GCNConv(dim_h, dim_h, **kwargs)
        self.conv4 = GCNConv(dim_h, dim_h, **kwargs)
        self.conv5 = GCNConv(dim_h, dim_h, **kwargs)

        # Linear layer
        self.lin1 = Linear(dim_h * 5, dim_h * 5)

        # Classification head
        self.lin2 = Linear(dim_h * 5, final_dim)

    def forward(self, x, edge_index, batch):
        # Embedding
        h1 = self.conv1(x, edge_index)
        h1 = h1.relu()
        h2 = self.conv2(h1, edge_index)
        h2 = h2.relu()
        h3 = self.conv3(h2, edge_index)
        h3 = h3.relu()
        h4 = self.conv4(h3, edge_index)
        h4 = h4.relu()
        h5 = self.conv4(h4, edge_index)
        h5 = h5.relu()

        # Graph-level readout
        h1 = global_mean_pool(h1, batch)
        h2 = global_mean_pool(h2, batch)
        h3 = global_mean_pool(h3, batch)
        h4 = global_mean_pool(h4, batch)
        h5 = global_mean_pool(h5, batch)

        h = torch.cat((h1, h2, h3, h4, h5), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)

        # return F.log_softmax(h, dim=1)
        return h.flatten()

###############


#### - GIN - ###
class GINModel(torch.nn.Module):
    def __init__(self, input_dim, dim_h, final_dim, **kwargs):
        super().__init__()
        torch.manual_seed(42)

        # Layers
        self.conv1 = GINConv(
            Sequential(Linear(input_dim, dim_h),
                       BatchNorm1d(dim_h),
                       ReLU(),
                       Linear(dim_h, dim_h),
                       ReLU()),
            **kwargs
        )
        self.conv2 = GINConv(
            Sequential(Linear(dim_h, dim_h),
                       BatchNorm1d(dim_h),
                       ReLU(),
                       Linear(dim_h, dim_h),
                       ReLU()),
            **kwargs
        )
        self.conv3 = GINConv(
            Sequential(Linear(dim_h, dim_h),
                       BatchNorm1d(dim_h),
                       ReLU(),
                       Linear(dim_h, dim_h),
                       ReLU()),
            **kwargs
        )
        self.conv4 = GINConv(
            Sequential(Linear(dim_h, dim_h),
                       BatchNorm1d(dim_h),
                       ReLU(),
                       Linear(dim_h, dim_h),
                       ReLU()),
            **kwargs
        )
        self.conv5 = GINConv(
            Sequential(Linear(dim_h, dim_h),
                       BatchNorm1d(dim_h),
                       ReLU(),
                       Linear(dim_h, dim_h),
                       ReLU()),
            **kwargs
        )

        # Linear layer
        self.lin1 = Linear(dim_h * 5, dim_h * 5)

        # Classification head
        self.lin2 = Linear(dim_h * 5, final_dim)

    def forward(self, x, edge_index, batch):
        # Embedding
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)
        h4 = self.conv4(h3, edge_index)
        h5 = self.conv5(h4, edge_index)

        # Graph-level readout
        h1 = global_mean_pool(h1, batch)
        h2 = global_mean_pool(h2, batch)
        h3 = global_mean_pool(h3, batch)
        h4 = global_mean_pool(h4, batch)
        h5 = global_mean_pool(h5, batch)

        h = torch.cat((h1, h2, h3, h4, h5), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)

        # return F.log_softmax(h, dim=1)
        return h.flatten()

###############


#### - GIN - ###
class GINEModel(torch.nn.Module):
    def __init__(self, input_dim, dim_h, final_dim, edge_dim, **kwargs):
        super().__init__()
        torch.manual_seed(42)

        # Layers
        self.conv1 = GINEConv(
            Sequential(Linear(input_dim, dim_h),
                       BatchNorm1d(dim_h),
                       ReLU(),
                       Linear(dim_h, dim_h),
                       ReLU()),
            edge_dim=edge_dim,
            **kwargs
        )
        self.conv2 = GINEConv(
            Sequential(Linear(dim_h, dim_h),
                       BatchNorm1d(dim_h),
                       ReLU(),
                       Linear(dim_h, dim_h),
                       ReLU()),
            edge_dim=edge_dim,
            **kwargs
        )
        self.conv3 = GINEConv(
            Sequential(Linear(dim_h, dim_h),
                       BatchNorm1d(dim_h),
                       ReLU(),
                       Linear(dim_h, dim_h),
                       ReLU()),
            edge_dim=edge_dim,
            **kwargs
        )
        self.conv4 = GINEConv(
            Sequential(Linear(dim_h, dim_h),
                        BatchNorm1d(dim_h),
                        ReLU(),
                        Linear(dim_h, dim_h),
                        ReLU()),
            edge_dim=edge_dim,
            **kwargs
        ) 
        self.conv5 = GINEConv(
            Sequential(Linear(dim_h, dim_h),
                        BatchNorm1d(dim_h),
                        ReLU(),
                        Linear(dim_h, dim_h),
                        ReLU()),
            edge_dim=edge_dim,
            **kwargs
        )

        # Linear layer
        self.lin1 = Linear(dim_h * 5, dim_h * 5)

        # Classification head
        self.lin2 = Linear(dim_h * 5, final_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        # Embedding
        h1 = self.conv1(x, edge_index, edge_attr)
        h2 = self.conv2(h1, edge_index, edge_attr)
        h3 = self.conv3(h2, edge_index, edge_attr)
        h4 = self.conv4(h3, edge_index, edge_attr)
        h5 = self.conv5(h4, edge_index, edge_attr)

        # Graph-level readout
        h1 = global_mean_pool(h1, batch)
        h2 = global_mean_pool(h2, batch)
        h3 = global_mean_pool(h3, batch)
        h4 = global_mean_pool(h4, batch)
        h5 = global_mean_pool(h5, batch)

        h = torch.cat((h1, h2, h3, h4, h5), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)

        # return F.log_softmax(h, dim=1)
        return h.flatten()

###############


################### - Functions - #############################################################
def get_all_models():
    """Get all available model blueprints"""

    model_blueprints = [
        EINModel,
        EINModel_v2,
        EINModel_v3,
        # EINModel_v4,
        EINModel_v5,
        GCNModel,
        GATModel,
        GINModel,
        GINEModel
    ]

    return model_blueprints
