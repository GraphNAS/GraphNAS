import torch.nn.functional as F

from models.operators import *


class GraphNet(torch.nn.Module):

    def __init__(self, actions, num_feat, num_label, drop_out=0.6, multi_label=False, batch_normal=True, residual=True,
                 state_num=5):
        '''
        :param actions:
        :param multi_label:
        '''
        super(GraphNet, self).__init__()
        # args

        self.multi_label = multi_label
        self.num_feat = num_feat
        self.num_label = num_label
        self.dropout = drop_out
        self.residual = residual
        # check structure of GNN
        self.layer_nums = self.evalate_actions(actions, state_num)

        # layer module
        self.build_model(actions, batch_normal, drop_out, num_feat, num_label, state_num)

    def build_model(self, actions, batch_normal, drop_out, num_feat, num_label, state_num):
        self.layers = torch.nn.ModuleList()
        self.gates = torch.nn.ModuleList()
        self.prediction = None
        self.build_hidden_layers(actions, batch_normal, drop_out, self.layer_nums, num_feat, num_label, state_num)

    def evalate_actions(self, actions, state_num):
        state_length = len(actions)
        if state_length % state_num != 0:
            raise RuntimeError("Wrong Input: unmatchable input")
        layer_nums = state_length // state_num
        if self.evaluate_structure(actions, layer_nums, state_num=state_num):
            pass
        else:
            raise RuntimeError("wrong structure")
        return layer_nums

    def evaluate_structure(self, actions, layer_nums, state_num=6):
        hidden_units_list = []
        out_channels_list = []
        for i in range(layer_nums):
            head_num = actions[i * state_num + 3]
            out_channels = actions[i * state_num + 4]
            hidden_units_list.append(head_num * out_channels)
            out_channels_list.append(out_channels)

        return out_channels_list[-1] == self.num_label

    def build_hidden_layers(self, actions, batch_normal, drop_out, layer_nums, num_feat, num_label, state_num=6):

        # build hidden layer
        for i in range(layer_nums):

            # compute input
            if i == 0:
                in_channels = num_feat
            else:
                in_channels = out_channels * head_num

            # extract operator types from action
            attention_type = actions[i * state_num + 0]
            aggregator_type = actions[i * state_num + 1]
            act = actions[i * state_num + 2]
            head_num = actions[i * state_num + 3]
            out_channels = actions[i * state_num + 4]
            # Multi-head used in GAT.
            # "concat" is True, concat output of each head;
            # "concat" is False, get average of each head output;
            concat = True
            if i == layer_nums - 1:
                concat = False  # The last layer get average
            else:
                pass

            if i == 0:
                residual = False and self.residual  # special setting of dgl
            else:
                residual = True and self.residual
            self.layers.append(
                NASLayer(attention_type, aggregator_type, act, head_num, in_channels, out_channels, dropout=drop_out,
                         concat=concat, residual=residual, batch_normal=batch_normal))

    def forward(self, feat, g):

        output = feat
        for i, layer in enumerate(self.layers):
            output = layer(output, g)

        return output

    def __repr__(self):
        result_lines = ""
        for each in self.layers:
            result_lines += str(each)
        return result_lines

    # map GNN's parameters into dict
    def get_param_dict(self, old_param=None, update_all=True):
        if old_param is None:
            result = {}
        else:
            result = old_param
        for i in range(self.layer_nums):
            key = "layer_%d" % i
            new_param = self.layers[i].get_param_dict()
            if key in result:
                new_param = NASLayer.merge_param(result[key], new_param, update_all)
                result[key] = new_param
            else:
                result[key] = new_param
        return result

    # load parameters from parameter dict
    def load_param(self, param):
        if param is None:
            return
        for i in range(self.layer_nums):
            self.layers[i].load_param(param["layer_%d" % i])


############################
#  Each layer of GNN
############################

def gat_message(edges):
    if 'norm' in edges.src:
        msg = edges.src['ft'] * edges.src['norm']
        return {'ft': edges.src['ft'], 'a2': edges.src['a2'], 'a1': edges.src['a1'], 'norm': msg}
    return {'ft': edges.src['ft'], 'a2': edges.src['a2'], 'a1': edges.src['a1']}


class NASLayer(nn.Module):
    def __init__(self, attention_type, aggregator_type, act, head_num, in_channels, out_channels=8, concat=True,
                 dropout=0.6, pooling_dim=128, residual=False, batch_normal=True):
        '''
        build one layer of GNN
        :param attention_type:
        :param aggregator_type:
        :param act: Activation function
        :param head_num: head num, in another word repeat time of current ops
        :param in_channels: input dimension
        :param out_channels: output dimension
        :param concat: concat output. get average when concat is False
        :param dropout: dropput for current layer
        :param pooling_dim: hidden layer dimension; set for pooling aggregator
        :param residual: whether current layer has  skip-connection
        :param batch_normal: whether current layer need batch_normal
        '''
        super(NASLayer, self).__init__()
        # print("NASLayer", in_channels, concat, residual)
        self.attention_type = attention_type
        self.aggregator_type = aggregator_type
        self.act = NASLayer.act_map(act)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = int(head_num)
        self.concat = concat
        self.dropout = dropout
        self.attention_dim = 1
        self.pooling_dim = pooling_dim

        self.batch_normal = batch_normal

        if attention_type in ['cos', 'generalized_linear']:
            self.attention_dim = 64
        self.bn = nn.BatchNorm1d(self.in_channels, momentum=0.5)
        self.prp = nn.ModuleList()
        self.red = nn.ModuleList()
        self.fnl = nn.ModuleList()
        self.agg = nn.ModuleList()
        for hid in range(self.num_heads):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.prp.append(AttentionPrepare(in_channels, out_channels, self.attention_dim, dropout))
            agg = NASLayer.aggregator_map(aggregator_type, out_channels, pooling_dim)
            self.agg.append(agg)
            self.red.append(NASLayer.attention_map(attention_type, dropout, agg, self.attention_dim))
            self.fnl.append(GATFinalize(hid, in_channels,
                                        out_channels, NASLayer.act_map(act), residual))

    @staticmethod
    def aggregator_map(aggregator_type, in_dim, pooling_dim):
        if aggregator_type == "sum":
            return SumAggregator()
        elif aggregator_type == "mean":
            return MeanPoolingAggregator(in_dim, pooling_dim)
        elif aggregator_type == "max":
            return MaxPoolingAggregator(in_dim, pooling_dim)
        elif aggregator_type == "mlp":
            return MLPAggregator(in_dim, pooling_dim)
        elif aggregator_type == "lstm":
            return LSTMAggregator(in_dim, pooling_dim)
        elif aggregator_type == "gru":
            return GRUAggregator(in_dim, pooling_dim)
        else:
            raise Exception("wrong aggregator type", aggregator_type)

    @staticmethod
    def attention_map(attention_type, attn_drop, aggregator, attention_dim):
        if attention_type == "gat":
            return GATReduce(attn_drop, aggregator)
        elif attention_type == "cos":
            return CosReduce(attn_drop, aggregator)
        elif attention_type == "none":
            return ConstReduce(attn_drop, aggregator)
        elif attention_type == "gat_sym":
            return GatSymmetryReduce(attn_drop, aggregator)
        elif attention_type == "linear":
            return LinearReduce(attn_drop, aggregator)
        elif attention_type == "bilinear":
            return CosReduce(attn_drop, aggregator)
        elif attention_type == "generalized_linear":
            return GeneralizedLinearReduce(attn_drop, attention_dim, aggregator)
        elif attention_type == "gcn":
            return GCNReduce(attn_drop, aggregator)
        else:
            raise Exception("wrong attention type")

    @staticmethod
    def act_map(act):
        if act == "linear":
            return lambda x: x
        elif act == "elu":
            return F.elu
        elif act == "sigmoid":
            return torch.sigmoid
        elif act == "tanh":
            return torch.tanh
        elif act == "relu":
            return torch.nn.functional.relu
        elif act == "relu6":
            return torch.nn.functional.relu6
        elif act == "softplus":
            return torch.nn.functional.softplus
        elif act == "leaky_relu":
            return torch.nn.functional.leaky_relu
        else:
            raise Exception("wrong activate function")

    def get_param_dict(self):
        params = {}

        key = "%d_%d_%d_%s" % (self.in_channels, self.out_channels, self.num_heads, self.attention_type)
        prp_key = key + "_" + str(self.attention_dim) + "_prp"
        agg_key = key + "_" + str(self.pooling_dim) + "_" + self.aggregator_type
        fnl_key = key + "_fnl"
        bn_key = "%d_bn" % self.in_channels
        params[prp_key] = self.prp.state_dict()
        params[agg_key] = self.agg.state_dict()
        # params[key+"_"+self.attention_type] = self.red.state_dict()
        params[fnl_key] = self.fnl.state_dict()
        params[bn_key] = self.bn.state_dict()
        return params

    def load_param(self, param):
        key = "%d_%d_%d_%s" % (self.in_channels, self.out_channels, self.num_heads, self.attention_type)
        prp_key = key + "_" + str(self.attention_dim) + "_prp"
        agg_key = key + "_" + str(self.pooling_dim) + "_" + self.aggregator_type
        fnl_key = key + "_fnl"
        bn_key = "%d_bn" % self.in_channels
        if prp_key in param:
            self.prp.load_state_dict(param[prp_key])

        # red_key = key+"_"+self.attention_type
        if agg_key in param:
            self.agg.load_state_dict(param[agg_key])
            for i in range(self.num_heads):
                self.red[i].aggregator = self.agg[i]

        if fnl_key in param:
            self.fnl.load_state_dict(param[fnl_key])

        if bn_key in param:
            self.bn.load_state_dict(param[bn_key])

    @staticmethod
    def merge_param(old_param, new_param, update_all):
        for key in new_param:
            if update_all or key not in old_param:
                old_param[key] = new_param[key]
        return old_param

    def forward(self, features, g):
        if self.batch_normal:
            last = self.bn(features)
        else:
            last = features

        for hid in range(self.num_heads):
            i = hid
            # prepare
            g.ndata.update(self.prp[i](last))
            # message passing
            g.update_all(gat_message, self.red[i], self.fnl[i])
        # merge all the heads
        if not self.concat:
            output = g.pop_n_repr('head0')
            for hid in range(1, self.num_heads):
                output = torch.add(output, g.pop_n_repr('head%d' % hid))
            output = output / self.num_heads
        else:
            output = torch.cat(
                [g.pop_n_repr('head%d' % hid) for hid in range(self.num_heads)], dim=1)
        del last
        return output
