import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear, Mish
from torch_geometric.utils import dropout_edge
from torch_geometric.nn.conv import GENConv, PointNetConv
from torch_geometric.nn.models import MLP, DeepGCNLayer
from torch_geometric.nn.pool import global_mean_pool, global_max_pool, fps, radius, knn
from torch_geometric.nn.unpool import knn_interpolate


class GridEncoder(nn.Module):
    def __init__(self,
                 node_grid_dim,
                 node_grid_emb,
                 ):
        super(GridEncoder, self).__init__()

        self.node_grid_encoder = nn.Sequential(
            nn.Conv2d(node_grid_dim, node_grid_emb // 4, 3, 1, 1),
            nn.BatchNorm2d(node_grid_emb // 4),
            nn.Mish(),
            nn.Conv2d(node_grid_emb // 4, node_grid_emb // 2, 3, 1, 1),
            nn.BatchNorm2d(node_grid_emb // 2),
            nn.Mish(),
            nn.Conv2d(node_grid_emb // 2, node_grid_emb, 3, 1, 1),
            nn.BatchNorm2d(node_grid_emb),
            nn.Mish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1)
        )

    def forward(self, x):
        x = self.node_grid_encoder(x)

        return x


class GraphEncoder(nn.Module):
    def __init__(self,
                 hidden_channels,
                 graph_out_channels,
                 num_layers,
                 dropout
                 ):
        super(GraphEncoder, self).__init__()
        self.dropout = dropout

        self.layers = nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax', t=1.0,
                           learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = Mish()

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=dropout, ckpt_grad=(i % 3 == 0))
            self.layers.append(layer)

        self.lin = Linear(hidden_channels, graph_out_channels)

    def forward(self, x, edge_index, edge_attr, batch):
        edge_index, edge_mask = dropout_edge(edge_index, p=0.2, force_undirected=True, training=self.training)
        edge_attr = edge_attr[edge_mask]
        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        global_x = global_mean_pool(x, batch)

        return self.lin(x), self.lin(global_x)


class SAModule(nn.Module):
    def __init__(self, ratio, r, net, in_channels, out_channels):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(net, add_self_loops=False)
        self.lin = Linear(in_channels, out_channels)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        # row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64)
        row, col = knn(pos, pos[idx], self.r, batch, batch[idx])
        edge_index = torch.stack([col, row], dim=0)
        x_dst = x[idx] if x is not None else None
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        x = x + self.lin(x_dst)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(nn.Module):
    def __init__(self, net):
        super(GlobalSAModule, self).__init__()
        self.nn = net

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class FPModule(nn.Module):
    def __init__(self, k, net):
        super(FPModule, self).__init__()
        self.k = k
        self.nn = net

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


class PointEncoder(nn.Module):
    def __init__(self,
                 point_attr_dim,
                 point_out_channels,
                 sample_ratio,
                 knn_num,
                 neighbor_num
                 ):
        super(PointEncoder, self).__init__()

        self.sa_modules = nn.ModuleList([
            SAModule(sample_ratio[0],
                     knn_num[0],
                     MLP([point_attr_dim + 3, 64, 128]),
                     point_attr_dim,
                     128),
            SAModule(sample_ratio[1],
                     knn_num[1],
                     MLP([128 + 3, 128, 256]),
                     128,
                     256)
        ])

        self.global_sa_module = GlobalSAModule(MLP([256 + 3, 256, 512]))

        self.fp_modules = nn.ModuleList([
            FPModule(neighbor_num[0],
                     MLP([512 + 256, 512, 256])),
            FPModule(neighbor_num[1],
                     MLP([256 + 128, 256, 128])),
            FPModule(neighbor_num[2],
                     MLP([128 + 3, 128, 128, point_out_channels]))
        ])

    def forward(self, x, pos, batch):
        sa_out = (x, pos, batch)
        sa_outputs = [sa_out]

        for sa_module in self.sa_modules:
            sa_out = sa_module(*sa_out)
            sa_outputs.append(sa_out)

        sa_out = self.global_sa_module(*sa_out)

        fp_out = sa_out
        for fp_module, skip_sa_out in zip(self.fp_modules, reversed(sa_outputs)):
            fp_out = fp_module(*fp_out, *skip_sa_out)

        return fp_out[0]
