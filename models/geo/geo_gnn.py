import torch
import torch.nn.functional as F

import models.geo.utils as util
from models.geo.geo_layer import GeoLayer
from models.gnn import GraphNet as BaseNet


class GraphNet(BaseNet):
    '''
    do not contain jump knowledge layer
    '''

    def __init__(self, actions, num_feat, num_label, drop_out=0.6, multi_label=False, batch_normal=True, state_num=5,
                 residual=False):
        self.residual = residual
        super(GraphNet, self).__init__(actions, num_feat, num_label, drop_out, multi_label, batch_normal, residual,
                                       state_num)

    def build_model(self, actions, batch_normal, drop_out, num_feat, num_label, state_num):
        if self.residual:
            self.fcs = torch.nn.ModuleList()
        self.layers = torch.nn.ModuleList()
        self.acts = []
        self.gates = torch.nn.ModuleList()
        self.build_hidden_layers(actions, batch_normal, drop_out, self.layer_nums, num_feat, num_label, state_num)

    def build_hidden_layers(self, actions, batch_normal, drop_out, layer_nums, num_feat, num_label, state_num=6):
        num_all_out_channels = num_feat
        # build hidden layer
        for i in range(layer_nums):
            # 设置输入输出维度
            if i == 0:
                in_channels = num_feat
            else:
                in_channels = out_channels * head_num
                if self.jk_type == "layer_concat":  # layer_concat 将之前的输入都concat起来，作为下一层的输入
                    in_channels = num_all_out_channels
            # 提取参数
            attention_type = actions[i * state_num + 0]
            aggregator_type = actions[i * state_num + 1]
            act = actions[i * state_num + 2]
            head_num = actions[i * state_num + 3]
            out_channels = actions[i * state_num + 4]
            concat = True
            if i == layer_nums - 1:
                concat = False

            self.layers.append(
                GeoLayer(in_channels, out_channels, head_num, concat, dropout=self.dropout,
                         att_type=attention_type, agg_type=aggregator_type,))
            self.acts.append(util.act_map(act))
            if self.residual:
                if concat:
                    self.fcs.append(torch.nn.Linear(in_channels, out_channels * head_num))
                else:
                    self.fcs.append(torch.nn.Linear(in_channels, out_channels))

    def forward(self, x, edge_index_all):
        output = x
        if self.residual:
            for act, layer, fc in zip(self.acts, self.layers, self.fcs):
                # output = F.dropout(output, p=self.dropout, training=self.training)
                output = act(layer(output, edge_index_all) + fc(output))
        else:
            for act, layer in zip(self.acts, self.layers):
                output = F.dropout(output, p=self.dropout, training=self.training)
                output = act(layer(output, edge_index_all))
        if not self.multi_label:
            output = F.log_softmax(output, dim=1)
        return output

    def __repr__(self):
        result_lines = ""
        for each in self.layers:
            result_lines += str(each)
        return result_lines

