import torch
import torch.nn as nn
import torch.nn.functional as F

'''
    The whole process of each layer of GNN includes ibut not limited following action：
    1.Feature transform (Prepare)
    2.Correlation measure (Reduce)
    3.Aggregation ( Aggregator in Reduce )
    4.Residual connection (Finalize)
'''


############################################
# Prepare, set Attention Weight
############################################

class AttentionPrepare(nn.Module):
    '''
        Attention Prepare Layer
    '''

    def __init__(self, input_dim, hidden_dim, attention_dim, drop):
        super(AttentionPrepare, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim, bias=False)
        if drop:
            self.drop = nn.Dropout(drop)
        else:
            self.drop = 0
        self.attn_l = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.attn_r = nn.Linear(hidden_dim, attention_dim, bias=False)
        nn.init.xavier_normal_(self.fc.weight.data, gain=1.414)
        nn.init.xavier_normal_(self.attn_l.weight.data, gain=1.414)
        nn.init.xavier_normal_(self.attn_r.weight.data, gain=1.414)

    def forward(self, feats):
        h = feats
        if self.drop:
            h = self.drop(h)
        ft = self.fc(h)
        a1 = self.attn_l(ft)
        a2 = self.attn_r(ft)
        return {'h': h, 'ft': ft, 'a1': a1, 'a2': a2}


######################################################################
# Reduce, apply different attention action and execute aggregation
######################################################################
class GATReduce(nn.Module):
    def __init__(self, attn_drop, aggregator=None):
        super(GATReduce, self).__init__()
        if attn_drop:
            self.attn_drop = nn.Dropout(p=attn_drop)
        else:
            self.attn_drop = 0
        self.aggregator = aggregator

    def apply_agg(self, neighbor):
        if self.aggregator:
            return self.aggregator(neighbor)
        else:
            return torch.sum(neighbor, dim=1)

    def forward(self, nodes):
        a1 = torch.unsqueeze(nodes.data['a1'], 1)  # shape (B, 1, 1)
        a2 = nodes.mailbox['a2']  # shape (B, deg, 1)
        ft = nodes.mailbox['ft']  # shape (B, deg, D)
        # attention
        a = a1 + a2  # shape (B, deg, 1)
        a = a.sum(-1, keepdim=True)  # Just in case the dimension is not zero
        e = F.softmax(F.leaky_relu(a), dim=1)
        if self.attn_drop:
            e = self.attn_drop(e)
        return {'accum': self.apply_agg(e * ft)}  # shape (B, D)


class ConstReduce(GATReduce):
    '''
        Attention coefficient is 1
    '''

    def __init__(self, attn_drop, aggregator=None):
        super(ConstReduce, self).__init__(attn_drop, aggregator)

    def forward(self, nodes):
        ft = nodes.mailbox['ft']  # shape (B, deg, D)
        # attention
        if self.attn_drop:
            ft = self.attn_drop(ft)
        return {'accum': self.apply_agg(1 * ft)}  # shape (B, D)


class GCNReduce(GATReduce):
    '''
        Attention coefficient is 1
    '''

    def __init__(self, attn_drop, aggregator=None):
        super(GCNReduce, self).__init__(attn_drop, aggregator)

    def forward(self, nodes):
        if 'norm' not in nodes.data:
            raise Exception("Wrong Data, has no norm")
        self_norm = nodes.data['norm']
        self_norm = self_norm.unsqueeze(1)
        results = nodes.mailbox['norm'] * self_norm
        return {'accum': self.apply_agg(results)}  # shape (B, D)


class LinearReduce(GATReduce):
    '''
        equal to neighbor's self-attention
    '''

    def __init__(self, attn_drop, aggregator=None):
        super(LinearReduce, self).__init__(attn_drop, aggregator)

    def forward(self, nodes):
        ft = nodes.mailbox['ft']  # shape (B, deg, D)
        a2 = nodes.mailbox['a2']
        a2 = a2.sum(-1, keepdim=True)  # shape (B, deg, D)
        # attention
        e = F.softmax(torch.tanh(a2), dim=1)
        if self.attn_drop:
            e = self.attn_drop(e)
        return {'accum': self.apply_agg(e * ft)}  # shape (B, D)


class GatSymmetryReduce(GATReduce):
    '''
        gat Symmetry version ( Symmetry cannot be guaranteed after softmax)
    '''

    def __init__(self, attn_drop, aggregator=None):
        super(GatSymmetryReduce, self).__init__(attn_drop, aggregator)

    def forward(self, nodes):
        a1 = torch.unsqueeze(nodes.data['a1'], 1)  # shape (B, 1, 1)
        b1 = torch.unsqueeze(nodes.data['a2'], 1)  # shape (B, 1, 1)
        a2 = nodes.mailbox['a2']  # shape (B, deg, 1)
        b2 = nodes.mailbox['a1']  # shape (B, deg, 1)
        ft = nodes.mailbox['ft']  # shape (B, deg, D)
        # attention
        a = a1 + a2  # shape (B, deg, 1)
        b = b1 + b2  # different attention_weight
        a = a + b
        a = a.sum(-1, keepdim=True)  # Just in case the dimension is not zero
        e = F.softmax(F.leaky_relu(a + b), dim=1)
        if self.attn_drop:
            e = self.attn_drop(e)
        return {'accum': self.apply_agg(e * ft)}  # shape (B, D)


class CosReduce(GATReduce):
    '''
        used in Gaan
    '''

    def __init__(self, attn_drop, aggregator=None):
        super(CosReduce, self).__init__(attn_drop, aggregator)

    def forward(self, nodes):
        a1 = torch.unsqueeze(nodes.data['a1'], 1)  # shape (B, 1, 1)
        a2 = nodes.mailbox['a2']  # shape (B, deg, 1)
        ft = nodes.mailbox['ft']  # shape (B, deg, D)
        # attention
        a = a1 * a2
        a = a.sum(-1, keepdim=True)  # shape (B, deg, 1)
        e = F.softmax(F.leaky_relu(a), dim=1)
        if self.attn_drop:
            e = self.attn_drop(e)
        return {'accum': self.apply_agg(e * ft)}  # shape (B, D)


class GeneralizedLinearReduce(GATReduce):
    '''
        used in GeniePath
    '''

    def __init__(self, attn_drop, hidden_dim, aggregator=None):
        super(GeneralizedLinearReduce, self).__init__(attn_drop, aggregator)
        self.generalized_linear = nn.Linear(hidden_dim, 1, bias=False)
        if attn_drop:
            self.attn_drop = nn.Dropout(p=attn_drop)
        else:
            self.attn_drop = 0

    def forward(self, nodes):
        a1 = torch.unsqueeze(nodes.data['a1'], 1)  # shape (B, 1, 1)
        a2 = nodes.mailbox['a2']  # shape (B, deg, 1)
        ft = nodes.mailbox['ft']  # shape (B, deg, D)
        # attention
        a = a1 + a2
        a = torch.tanh(a)
        a = self.generalized_linear(a)
        e = F.softmax(a, dim=1)
        if self.attn_drop:
            e = self.attn_drop(e)
        return {'accum': self.apply_agg(e * ft)}  # shape (B, D)


#######################################################
# Aggregators, aggregate information from neighbor
#######################################################

class SumAggregator(nn.Module):

    def __init__(self):
        super(SumAggregator, self).__init__()

    def forward(self, neighbor):
        return torch.sum(neighbor, dim=1)


class MaxPoolingAggregator(SumAggregator):

    def __init__(self, input_dim, pooling_dim=512, num_fc=1, act=F.leaky_relu_):
        super(MaxPoolingAggregator, self).__init__()
        out_dim = input_dim
        self.fc = nn.ModuleList()
        self.act = act
        if num_fc > 0:
            for i in range(num_fc - 1):
                self.fc.append(nn.Linear(out_dim, pooling_dim))
                out_dim = pooling_dim
            self.fc.append(nn.Linear(out_dim, input_dim))

    def forward(self, ft):
        for layer in self.fc:
            ft = self.act(layer(ft))

        return torch.max(ft, dim=1)[0]


class MeanPoolingAggregator(MaxPoolingAggregator):

    def __init__(self, input_dim, pooling_dim=512, num_fc=1, act=F.leaky_relu_):
        super(MeanPoolingAggregator, self).__init__(input_dim, pooling_dim, num_fc, act)

    def forward(self, ft):
        for layer in self.fc:
            ft = self.act(layer(ft))

        return torch.mean(ft, dim=1)


class MLPAggregator(MaxPoolingAggregator):

    def __init__(self, input_dim, pooling_dim=512, num_fc=1, act=F.leaky_relu_):
        super(MLPAggregator, self).__init__(input_dim, pooling_dim, num_fc, act)

    def forward(self, ft):
        ft = torch.sum(ft, dim=1)
        for layer in self.fc:
            ft = self.act(layer(ft))
        return ft


class LSTMAggregator(SumAggregator):

    def __init__(self, input_dim, pooling_dim=512):
        super(LSTMAggregator, self).__init__()
        self.lstm = nn.LSTM(input_dim, pooling_dim, batch_first=True, bias=False)
        self.linear = nn.Linear(pooling_dim, input_dim)

    def forward(self, ft):
        torch.transpose(ft, 1, 0)
        hidden = self.lstm(ft)[0]
        return self.linear(torch.squeeze(hidden[-1], dim=0))


class GRUAggregator(SumAggregator):

    def __init__(self, input_dim, pooling_dim=512):
        super(LSTMAggregator, self).__init__()
        self.lstm = nn.GRU(input_dim, pooling_dim, batch_first=True, bias=False)
        self.linear = nn.Linear(pooling_dim, input_dim)

    def forward(self, ft):
        torch.transpose(ft, 1, 0)
        hidden = self.lstm(ft)[0]
        return self.linear(torch.squeeze(hidden[-1], dim=0))


######################################################################
# Finalize, introduce residual connection
######################################################################
class GATFinalize(nn.Module):
    '''
        concat + 全连接层
    '''

    def __init__(self, headid, indim, hiddendim, activation, residual):
        super(GATFinalize, self).__init__()
        self.headid = headid
        self.activation = activation
        self.residual = residual
        self.residual_fc = None
        if residual:
            if indim != hiddendim:
                self.residual_fc = nn.Linear(indim, hiddendim, bias=False)
                nn.init.xavier_normal_(self.residual_fc.weight.data, gain=1.414)

    def forward(self, nodes):
        ret = nodes.data['accum']
        if self.residual:
            if self.residual_fc is not None:
                ret = self.residual_fc(nodes.data['h']) + ret
            else:
                ret = nodes.data['h'] + ret
        return {'head%d' % self.headid: self.activation(ret)}
