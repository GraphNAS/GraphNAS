import torch

from graphnas_variants.macro_graphnas.pyg.pyg_gnn import GraphNet, act_map, F
from graphnas_variants.micro_graphnas.micro_search_space import gnn_map


class SimpleGNN(GraphNet):

    def build_model(self, actions, batch_normal, drop_out, num_feat, num_label, state_num=2):
        self.layers = torch.nn.ModuleList()
        self.acts = []
        self.build_hidden_layers(actions, batch_normal, drop_out, self.layer_nums, num_feat, num_label, state_num)

    def build_hidden_layers(self, actions, batch_normal, drop_out, layer_nums, num_feat, num_label, state_num=2):
        # build hidden layer
        for i in range(layer_nums):

            if i == 0:
                in_channels = num_feat
            else:
                in_channels = out_channels
            out_channels = actions[i]['out_dim']
            self.layers.append(gnn_map(actions[i]['conv_type'], in_channels, out_channels))
            self.acts.append(act_map("relu"))

    def forward(self, x, edge_index_all):
        output = x
        if self.residual:
            for i, (act, layer, fc) in enumerate(zip(self.acts, self.layers, self.fcs)):
                output = F.dropout(output, p=self.dropout, training=self.training)
                if self.batch_normal:
                    output = self.bns[i](output)

                output = act(layer(output, edge_index_all) + fc(output))
        else:
            for i, (act, layer) in enumerate(zip(self.acts, self.layers)):
                output = F.dropout(output, p=self.dropout, training=self.training)
                if self.batch_normal:
                    output = self.bns[i](output)
                output = act(layer(output, edge_index_all))
        if not self.multi_label:
            output = F.log_softmax(output, dim=1)
        return output

    def __repr__(self):
        result_lines = ""
        for each in self.layers:
            result_lines += str(each)
        return result_lines

    def get_param_dict(self, old_param=None, update_all=True):
        if old_param is None:
            result = {}
        else:
            result = old_param
        for i in range(self.layer_nums):
            key = "layer_%d" % i
            new_param = self.layers[i].get_param_dict()
            if key in result:
                new_param = self.merge_param(result[key], new_param, update_all)
                result[key] = new_param
            else:
                result[key] = new_param
        if self.residual:
            for i, fc in enumerate(self.fcs):
                key = f"layer_{i}_fc_{fc.weight.size(0)}_{fc.weight.size(1)}"
                result[key] = self.fcs[i]
        if self.batch_normal:
            for i, bn in enumerate(self.bns):
                key = f"layer_{i}_fc_{bn.weight.size(0)}"
                result[key] = self.bns[i]
        return result

    def load_param(self, param):
        if param is None:
            return

        for i in range(self.layer_nums):
            self.layers[i].load_param(param["layer_%d" % i])

        if self.residual:
            for i, fc in enumerate(self.fcs):
                key = f"layer_{i}_fc_{fc.weight.size(0)}_{fc.weight.size(1)}"
                if key in param:
                    self.fcs[i] = param[key]
        if self.batch_normal:
            for i, bn in enumerate(self.bns):
                key = f"layer_{i}_fc_{bn.weight.size(0)}"
                if key in param:
                    self.bns[i] = param[key]

    def evalate_actions(self, actions, state_num):
        layer_nums = len(actions)
        if actions[layer_nums - 1]["out_dim"] == self.num_label:
            pass
        else:
            raise RuntimeError("wrong structure")
        return layer_nums
