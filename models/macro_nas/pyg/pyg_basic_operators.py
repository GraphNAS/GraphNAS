import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax, scatter_


def att_map(att_name, heads, out_channels):
    if att_name == "gat":
        return GatAttention(heads, out_channels)
    elif att_name == "gat_sym":
        return GatSymAttention(heads, out_channels)
    elif att_name == "linear":
        return LinearAttention(heads, out_channels)
    elif att_name == "cos":
        return CosAttention(heads, out_channels)
    elif att_name == "generalized_linear":
        return GeneralizedLinearAttention(heads, out_channels)
    elif att_name == "common":
        return CommonAttention(heads, out_channels, 128, F.relu)
    elif att_name in ["const", "gcn"]:
        return ConstAttention()


def batch_agg_map(agg_type, in_channels, hidden_dim, num_head, dropout):
    if agg_type in ["add", "sum", "mean", "max"]:
        return BatchBasicAggregator(agg_type=agg_type, num_head=num_head, dropout=dropout)
    elif "pooling" in agg_type:
        agg_type = agg_type.split("_")[0]
        return BatchPoolingAggregator(agg_type=agg_type, in_channels=in_channels, hidden_dim=hidden_dim,
                                      num_head=num_head, dropout=dropout)
    elif agg_type in ["lstm"]:
        return LSTMAggregator(in_channels, hidden_dim, num_head, num_hidden_layers=1, dropout=dropout)
    elif agg_type in ["mlp"]:
        return BatchMLPAggregator(in_channels, hidden_dim, num_head, dropout)
    else:
        raise RuntimeError(f"wrong aggregate type:{agg_type}")


class ConstAttention(nn.Module):
    def __init__(self, **kwargs):
        super(ConstAttention, self).__init__()

    def forward(self, neighbor_vecs, self_vecs):
        # return torch.ones_like(neighbor_vecs).mean(dim=-1).to(neighbor_vecs.device)
        # size = neighbor_vecs.size()
        # return torch.ones([size[0], size[1], 1]).to(neighbor_vecs.device)
        return 1


class GatAttention(ConstAttention):
    def __init__(self, num_heads, out_channels):
        super(GatAttention, self).__init__()
        self.num_heads = num_heads
        # self.in_channels = in_channels
        self.out_channels = out_channels
        # self.att = Parameter(torch.Tensor(1, num_heads, 2 * out_channels))

        self.att_self_weight = Parameter(torch.Tensor(1, self.num_heads, self.out_channels))
        self.att_neighbor_weight = Parameter(torch.Tensor(1, self.num_heads, self.out_channels))
        # self.att_neighbor_weight = Parameter(self.in_channels, self.out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        # glorot(self.att_self_weight)
        # glorot(self.att_neighbor_weight)
        # glorot(self.att)
        pass

    def forward(self, neighbor_vecs, self_vecs):
        # shape [num_nodes, num_sample, num_heads]
        alpha = (self_vecs * self.att_self_weight).sum(dim=-1) + (neighbor_vecs * self.att_neighbor_weight).sum(dim=-1)
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        # alpha = torch.softmax(alpha, dim=-2)
        # Sample attention coefficients stochastically.
        return alpha


class GatSymAttention(GatAttention):

    def forward(self, neighbor_vecs, self_vecs):
        alpha = (self_vecs * self.att_self_weight).sum(dim=-1) + (neighbor_vecs * self.att_neighbor_weight).sum(dim=-1)
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha_2 = (neighbor_vecs * self.att_self_weight).sum(dim=-1) + (self_vecs * self.att_neighbor_weight).sum(
            dim=-1)
        alpha = alpha + alpha_2
        return alpha


class LinearAttention(GatAttention):
    def forward(self, neighbor_vecs, self_vecs):
        # self.att_self_weight = self.att[:, :, :self.out_channels]
        # self.att_neighbor_weight = self.att[:, :, self.out_channels:]

        al = neighbor_vecs * self.att_neighbor_weight
        ar = neighbor_vecs * self.att_self_weight
        alpha = al.sum(dim=-1) + ar.sum(dim=-1)
        alpha = torch.tanh(alpha)
        return alpha


class CosAttention(GatAttention):
    def forward(self, neighbor_vecs, self_vecs):
        alpha = neighbor_vecs * self.att_neighbor_weight * self_vecs * self.att_self_weight
        alpha = alpha.sum(dim=-1)
        return alpha


class GeneralizedLinearAttention(ConstAttention):
    def __init__(self, num_heads, out_channels):
        super(GeneralizedLinearAttention, self).__init__()
        self.num_heads = num_heads
        self.out_channels = out_channels

        self.att_self_weight = Parameter(torch.Tensor(1, self.num_heads, self.out_channels))
        self.att_neighbor_weight = Parameter(torch.Tensor(1, self.num_heads, self.out_channels))

        self.general_layer = nn.Linear(self.out_channels, 1)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_self_weight)
        glorot(self.att_neighbor_weight)
        glorot(self.general_layer.weight)
        zeros(self.general_layer.bias)

    def forward(self, neighbor_vecs, self_vecs):
        al = self_vecs * self.att_self_weight
        ar = neighbor_vecs * self.att_neighbor_weight
        alpha = al + ar
        alpha = torch.tanh(alpha)
        alpha = self.general_layer(alpha)
        return alpha


class CommonAttention(ConstAttention):
    def __init__(self, num_heads, out_channels, hidden_dim, act=lambda x: x):
        super(CommonAttention, self).__init__()
        self.num_heads = num_heads
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.act = act

        self.att_self_weight = nn.ModuleList()
        self.att_neighbor_weight = nn.ModuleList()

        for i in range(num_heads):
            self.att_self_weight.append(nn.Linear(out_channels, hidden_dim))
            self.att_neighbor_weight.append(nn.Linear(out_channels, hidden_dim))

        self.general_layer = nn.Linear(self.hidden_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        for each in self.att_self_weight:
            glorot(each.weight)
            zeros(each.bias)
        for each in self.att_neighbor_weight:
            glorot(each.weight)
            zeros(each.bias)
        glorot(self.general_layer.weight)
        zeros(self.general_layer.bias)

    def forward(self, neighbor_vecs, self_vecs):
        alpha = []
        for i in range(self.num_heads):
            self_weight = self.att_self_weight[i]
            neight_weight = self.att_neighbor_weight[i]
            tmp_self_vecs = self_weight(self_vecs[:, i, :])
            tmp_neight_vecs = neight_weight(neighbor_vecs[:, i, :])
            alpha.append(self.general_layer(self.act(tmp_self_vecs + tmp_neight_vecs)))
        return torch.cat(alpha, dim=-1)


class BasicAggregator(nn.Module):
    def __init__(self, agg_type="add", num_head=1, dropout=0.6):
        super(BasicAggregator, self).__init__()
        if agg_type == "sum":
            agg_type = "add"
        self.agg_type = agg_type
        self.dropout = dropout
        self.num_head = num_head
        assert self.agg_type in ['add', 'mean', 'max']

    def forward(self, neighbor_vecs, alpha, edge_index, num_nodes):
        neighbor = self.preprocess(alpha, edge_index, neighbor_vecs, num_nodes)
        out = scatter_(self.agg_type, neighbor, edge_index[1], dim_size=num_nodes)
        return out

    def preprocess(self, alpha, edge_index, neighbor_vecs, num_nodes):
        if isinstance(alpha, int):
            if self.training and self.dropout > 0:
                neighbor_vecs = F.dropout(neighbor_vecs, p=self.dropout, training=self.training)
            return alpha*neighbor_vecs
        else:
            alpha = softmax(alpha, edge_index[0], num_nodes)
            # Sample attention coefficients stochastically.
            if self.training and self.dropout > 0:
                alpha = F.dropout(alpha, p=self.dropout, training=self.training)
            neighbor = neighbor_vecs * alpha.view(-1, self.num_head, 1)
        return neighbor


class PoolingAggregator(BasicAggregator):
    def __init__(self, in_channels, hidden_dim=128, agg_type="add", num_head=1, dropout=0.6):
        super(PoolingAggregator, self).__init__(agg_type, num_head, dropout)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_channels, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, in_channels))

    def forward(self, neighbor_vecs, alpha, edge_index, num_nodes):
        neighbor = self.preprocess(alpha, edge_index, neighbor_vecs, num_nodes)
        for layer in self.layers:
            neighbor_vecs = layer(neighbor_vecs)
        out = scatter_(self.agg_type, neighbor, edge_index[0], dim_size=num_nodes)
        return out


class BatchBasicAggregator(nn.Module):
    def __init__(self, agg_type="add", num_head=1, dropout=0.6):
        super(BatchBasicAggregator, self).__init__()
        self.agg_type = agg_type
        self.dropout = dropout
        self.num_head = num_head
        assert self.agg_type in ['add', "sum", 'mean', 'max']

    def forward(self, neighbor_vecs, alpha, num_sample, need_softmax=False):
        out = self.preprocess(alpha, need_softmax, neighbor_vecs, num_sample)
        if self.agg_type in ["add", "sum"]:
            out = out.sum(dim=-3)
        elif self.agg_type in ["mean"]:
            out = out.mean(dim=-3)
        elif self.agg_type in ["max"]:
            out = out.max(dim=-3)[0]
        return out

    def preprocess(self, alpha, need_softmax, neighbor_vecs, num_sample):
        # shape [num_nodes, num_sample, num_heads]
        if isinstance(alpha, torch.Tensor):
            if need_softmax:
                alpha = torch.softmax(alpha, dim=-2)
            # Sample attention coefficients stochastically.
            if self.training and self.dropout > 0:
                alpha = F.dropout(alpha, p=self.dropout, training=self.training)
            out = neighbor_vecs * alpha.view(-1, num_sample, self.num_head, 1)
        else:
            out = neighbor_vecs * alpha
        return out


class BatchPoolingAggregator(BatchBasicAggregator):
    def __init__(self, in_channels, hidden_dim=128, num_head=1, dropout=0.6, agg_type="add"):
        super(BatchPoolingAggregator, self).__init__(agg_type, num_head, dropout)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_channels, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, in_channels))

    def reset_parameters(self):
        for layer in self.layers:
            glorot(layer.weight)
            zeros(layer.bias)

    def forward(self, neighbor_vecs, alpha, num_sample, need_softmax=False):
        out = self.preprocess(alpha, need_softmax, neighbor_vecs, num_sample)
        # print(out.sum())
        # print(neighbor_vecs.sum())
        for layer in self.layers:
            out = layer(out)
            # neighbor_vecs = layer(neighbor_vecs)
        out = neighbor_vecs
        if self.agg_type in ["add", "sum"]:
            out = out.sum(dim=-3)
        elif self.agg_type in ["mean"]:
            out = out.mean(dim=-3)
        elif self.agg_type in ["max"]:
            out = out.max(dim=-3)[0]
        return out


class BatchMLPAggregator(BatchPoolingAggregator):
    def __init__(self, in_channels, hidden_dim=128, num_head=1, dropout=0.6):
        super(BatchMLPAggregator, self).__init__(in_channels, hidden_dim, num_head, dropout, "add")

    def forward(self, neighbor_vecs, alpha, num_sample, need_softmax=False):
        out = self.preprocess(alpha, need_softmax, neighbor_vecs, num_sample)
        out = out.sum(dim=1)
        for layer in self.layers:
            out = layer(out)

        return out


class LSTMAggregator(BatchBasicAggregator):
    '''
    multi-head share the same aggregator
    '''

    def __init__(self, in_channels, hidden_dim=128, num_head=1, num_hidden_layers=1, dropout=0.6):
        super(LSTMAggregator, self).__init__(num_head=num_head, dropout=dropout)
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_head = num_head
        self.num_hidden_layers = num_hidden_layers
        assert self.agg_type in ['add', 'mean', 'max']
        self.lstm = nn.LSTM(in_channels * self.num_head, hidden_dim * self.num_head, num_hidden_layers,
                            batch_first=True)
        # parameters are reset in LSTM

    def forward(self, neighbor_vecs, alpha, num_sample, need_softmax=False):
        out = self.preprocess(alpha, need_softmax, neighbor_vecs, num_sample)

        # apply aggregator
        out = out.view(-1, num_sample, self.in_channels * self.num_head)
        batch_size = out.size(0)
        h0 = torch.zeros(self.num_hidden_layers, batch_size, self.hidden_dim * self.num_head).to(neighbor_vecs.device)
        c0 = torch.zeros(self.num_hidden_layers, batch_size, self.hidden_dim * self.num_head).to(neighbor_vecs.device)
        output, _ = self.lstm(out, (h0, c0))

        out = output[:, -1, :]
        out = out.view(batch_size, self.num_head, self.hidden_dim)
        return out
