import json
from copy import deepcopy

import pandas as pd
from matplotlib import pyplot as plt


class Analysis:
    def __init__(self, link, type, dataset, nr_layers):
        self.link = link
        self.type = type
        self.dataset = dataset
        self.nr_layers = nr_layers

        if self.type == 'macro':
            self.df = self.return_df_macro()
        elif self.type == 'micro':
            self.hp_list = ['Learning_Rate', 'Dropout', 'L2', 'Hidden_Layer_Size']
            self.df = self.return_df_micro()
        else:
            raise Exception("marco/micro")

    def return_df_macro(self):
        f = self.read_file_macro()
        df = pd.DataFrame(f, columns=['actions', 'val'])
        df[['layer_1', 'agg_1', 'acc_1', 'att_1', 'hidd_dim_1', 'layer_2', 'agg_2', 'acc_2', 'att_2',
            'hidd_dim_2']] = pd.DataFrame(df.actions.tolist(), index=df.index)
        return df

    def return_df_micro(self):
        f = self.read_file_micro()
        df = pd.DataFrame(f, columns=['actions', 'hyper_param', 'val'])
        df[['in_1', 'f_1', 'in_2', 'f_2', 'act', 'agg']] = pd.DataFrame(df.actions.tolist(), index=df.index)
        df[self.hp_list] = pd.DataFrame(df.hyper_param.tolist(),
                                                                                   index=df.index)
        return df

    def read_file_macro(self):
        results = []
        with open(self.link, "r") as f:
            lines = f.readlines()

        for line in lines:
            actions = line[:line.index(";")]

            actions_dataform = str(actions).strip("'<>() ").replace('\'', '\"')
            actions_struct = {}
            actions_struct = json.loads(actions_dataform)

            val_score = line.split(";")[-1]
            val_dataform = str(val_score).strip("'<>() ").replace('\'', '\"')
            val_struct = {}
            val_struct = json.loads(val_dataform)

            results.append((actions_struct, val_struct))

        return results

    def read_file_micro(self):
        results = []
        with open(self.link, "r") as f:
            lines = f.readlines()

        for line in lines:
            actions = line[:line.index(";")]

            dataform = str(actions).strip("'<>() ").replace('\'', '\"')
            struct = {}
            struct = json.loads(dataform)
            actions_struct = struct['action']
            hyper_param_struct = struct['hyper_param']

            val_score = line.split(";")[-1]
            val_dataform = str(val_score).strip("'<>() ").replace('\'', '\"')
            val_struct = {}
            val_struct = json.loads(val_dataform)

            results.append((actions_struct, hyper_param_struct, val_struct))

        return results

    def analise_over_action(self, action_name):
        new_df = pd.DataFrame(self.df[[action_name, 'val']].groupby(action_name).mean())
        return new_df

    def analise_over_combinations_of_actions(self, actions):
        l = deepcopy(actions)
        actions.append('val')
        print(l)
        df = self.df[actions].groupby(l).mean()
        return df

    def explore_hyper_param(self):
        if self.type != 'micro':
            raise Exception("Sorry, wrong type")

        figure, axis = plt.subplots(1, len(self.hp_list), figsize=(7, 4))
        for i in range(len(self.hp_list)):
            A = self.analise_over_action(self.hp_list[i])
            x = A.index
            y = A['val']

            x_pos = [i for i, _ in enumerate(x)]

            axis[i].bar(x_pos, y)
            axis[i].set_ylabel("Validation Accuracy")
            axis[i].set_xlabel(self.hp_list[i])
            axis[i].set_ylim(0.45, 0.9)
            #axis[i].set_title("Hyper Parameter Analysis over " + self.dataset + ' GraphNAS '+ self.type)
            axis[i].set_xticklabels(x, fontdict=None, minor=False)

        for ax in figure.get_axes():
            ax.label_outer()
        figure.title("Hyper Parameter Analysis over " + self.dataset + ' GraphNAS '+ self.type)
        plt.show()

    def explore_micro_architectures(self):
        if self.type != 'micro':
            raise Exception("Sorry, wrong type")


cora_micro = Analysis('../Cora_microsub_manager_logger_file_1672948551.2461584.txt', dataset='Cora', type='micro', nr_layers = 2)
cora_micro.explore_hyper_param()
