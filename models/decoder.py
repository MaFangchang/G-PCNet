import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn.models import MLP
from torch_scatter import scatter_mean


def instance_voter(inst_out):
    def process_tensor(tensor):
        blocks = tensor.unfold(0, 25, 25).unfold(1, 25, 25)
        means = blocks.mean(dim=(2, 3))
        return means

    result_tensors = [process_tensor(inst) for inst in inst_out]
    result = torch.stack(result_tensors)

    return result


class SemanticHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SemanticHead, self).__init__()
        self.mlp = MLP([in_channels, in_channels * 2, num_classes], dropout=0.1)

    def forward(self, x, point_batches):
        x = self.mlp(x)

        if not self.training:
            x = scatter_mean(x, point_batches, dim=0)

        return x


class BottomHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BottomHead, self).__init__()
        self.mlp = MLP([in_channels, in_channels * 2, out_channels], dropout=0.1)

    def forward(self, x, point_batches):
        x = self.mlp(x)

        if not self.training:
            x = scatter_mean(x, point_batches, dim=0)

        return x


class InstanceHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InstanceHead, self).__init__()
        self.Wq = MLP([in_channels, in_channels * 2, out_channels], dropout=0.1, norm='layer_norm')
        self.Wk = MLP([in_channels, in_channels * 2, out_channels], dropout=0.1, norm='layer_norm')

    def forward(self, x, f_point_batches, g_node_batches):
        x = scatter_mean(x, f_point_batches, dim=0)

        _, counts = torch.unique(g_node_batches, return_counts=True)
        batch_num_nodes = counts.tolist()
        hidden_list = list(torch.split(x, batch_num_nodes, dim=0))
        padded_hidden = pad_sequence(hidden_list, batch_first=True)
        q = self.Wq(padded_hidden)
        k = self.Wk(padded_hidden)
        inst_out = torch.bmm(q, k.transpose(1, 2))

        return inst_out
