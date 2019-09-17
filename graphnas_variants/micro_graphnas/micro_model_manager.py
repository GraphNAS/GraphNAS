from graphnas_variants.macro_graphnas.pyg.pyg_gnn_model_manager import GeoCitationManager
from graphnas_variants.micro_graphnas.micro_gnn import MicroGNN
import torch
import os.path as osp
from torch_geometric.datasets import Planetoid, Coauthor, Amazon
import torch_geometric.transforms as T
from graphnas.utils.label_split import fix_size_split


def load_data(dataset="Cora", supervised=False, full_data=True):
    '''
    support semi-supervised and supervised
    :param dataset:
    :param supervised:
    :return:
    '''
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
    if dataset in ["CS", "Physics"]:
        dataset = Coauthor(path, dataset, T.NormalizeFeatures())
    elif dataset in ["Computers", "Photo"]:
        dataset = Amazon(path,dataset, T.NormalizeFeatures())
    elif dataset in ["Cora", "Citeseer", "Pubmed"]:
        dataset = Planetoid(path, dataset, T.NormalizeFeatures())
    data = dataset[0]
    if supervised:
        if full_data:
            data.train_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
            data.train_mask[:-1000] = 1
            data.val_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
            data.val_mask[-1000: -500] = 1
            data.test_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
            data.test_mask[-500:] = 1
        else:
            data.train_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
            data.train_mask[:1000] = 1
            data.val_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
            data.val_mask[1000: 1500] = 1
            data.test_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
            data.test_mask[1500:2000] = 1
    return data


class MicroCitationManager(GeoCitationManager):

    def __init__(self, args):
        super(MicroCitationManager, self).__init__(args)
        if hasattr(args, "supervised")  :
            self.data = load_data(args.dataset, args.supervised)
            device = torch.device('cuda' if args.cuda else 'cpu')
            self.data.to(device)

    def build_gnn(self, actions):
        model = MicroGNN(actions, self.in_feats, self.n_classes, layers=self.args.layers_of_child_model, num_hidden=self.args.num_hidden,
                         dropout=self.args.in_drop)
        return model

    def train(self, actions=None, format="micro"):
        self.current_action = actions
        print(actions)
        model_actions = actions['action']
        param = actions['hyper_param']
        self.args.lr = param[0]
        self.args.in_drop = param[1]
        self.args.weight_decay = param[2]
        self.args.num_hidden = param[3]
        return super(GeoCitationManager, self).train(model_actions, format=format)

    def record_action_info(self, origin_action, reward, val_acc):
        return super(GeoCitationManager, self).record_action_info(self.current_action, reward, val_acc)

    def evaluate(self, actions=None, format="micro"):
        print(actions)
        model_actions = actions['action']
        param = actions['hyper_param']
        self.args.lr = param[0]
        self.args.in_drop = param[1]
        self.args.weight_decay = param[2]
        self.args.num_hidden = param[3]
        return super(GeoCitationManager, self).evaluate(model_actions, format=format)

    def shuffle_data(self, full_data=True):
        device = torch.device('cuda' if self.args.cuda else 'cpu')
        if full_data:
            self.data = fix_size_split(self.data, self.data.num_nodes-1000, 500, 500)
        else:
            self.data = fix_size_split(self.data, 1000, 500, 500)
        self.data.to(device)