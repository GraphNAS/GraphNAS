import os.path as osp
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Coauthor, Amazon

from graphnas.gnn_model_manager import CitationGNNManager, evaluate
from graphnas_variants.macro_graphnas.pyg.pyg_gnn import GraphNet
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
        dataset = Amazon(path, dataset, T.NormalizeFeatures())
    elif dataset in ["Cora", "Citeseer", "Pubmed"]:
        dataset = Planetoid(path, dataset, T.NormalizeFeatures())
    data = dataset[0]
    if supervised:
        if full_data:
            data.train_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
            data.train_mask[:-1000] = 1
            data.val_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
            data.val_mask[data.num_nodes - 1000: data.num_nodes - 500] = 1
            data.test_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
            data.test_mask[data.num_nodes - 500:] = 1
        else:
            data.train_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
            data.train_mask[:1000] = 1
            data.val_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
            data.val_mask[data.num_nodes - 1000: data.num_nodes - 500] = 1
            data.test_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
            data.test_mask[data.num_nodes - 500:] = 1
    return data


class GeoCitationManager(CitationGNNManager):
    def __init__(self, args):
        super(GeoCitationManager, self).__init__(args)
        if hasattr(args, "supervised"):
            self.data = load_data(args.dataset, args.supervised)
        else:
            self.data = load_data(args.dataset)
        self.args.in_feats = self.in_feats = self.data.num_features
        self.args.num_class = self.n_classes = self.data.y.max().item() + 1
        device = torch.device('cuda' if args.cuda else 'cpu')
        self.data.to(device)

    def build_gnn(self, actions):
        model = GraphNet(actions, self.in_feats, self.n_classes, drop_out=self.args.in_drop, multi_label=False,
                         batch_normal=False, residual=False)
        return model

    def update_args(self, args):
        self.args = args

    def save_param(self, model, update_all=False):
        pass

    def shuffle_data(self, full_data=True):
        device = torch.device('cuda' if self.args.cuda else 'cpu')
        if full_data:
            self.data = fix_size_split(self.data, self.data.num_nodes - 1000, 500, 500)
        else:
            self.data = fix_size_split(self.data, 1000, 500, 500)
        self.data.to(device)

    @staticmethod
    def run_model(model, optimizer, loss_fn, data, epochs, early_stop=5, tmp_model_file="geo_citation.pkl",
                  half_stop_score=0, return_best=False, cuda=True, need_early_stop=False, show_info=False):

        dur = []
        begin_time = time.time()
        best_performance = 0
        min_val_loss = float("inf")
        min_train_loss = float("inf")
        model_val_acc = 0
        print("Number of train datas:", data.train_mask.sum())
        for epoch in range(1, epochs + 1):
            model.train()
            t0 = time.time()
            # forward
            logits = model(data.x, data.edge_index)
            logits = F.log_softmax(logits, 1)
            loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = loss.item()

            # evaluate
            model.eval()
            logits = model(data.x, data.edge_index)
            logits = F.log_softmax(logits, 1)
            train_acc = evaluate(logits, data.y, data.train_mask)
            dur.append(time.time() - t0)

            val_acc = evaluate(logits, data.y, data.val_mask)
            test_acc = evaluate(logits, data.y, data.test_mask)

            loss = loss_fn(logits[data.val_mask], data.y[data.val_mask])
            val_loss = loss.item()
            if val_loss < min_val_loss:  # and train_loss < min_train_loss
                min_val_loss = val_loss
                min_train_loss = train_loss
                model_val_acc = val_acc
                if test_acc > best_performance:
                    best_performance = test_acc
            if show_info:
                print(
                    "Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f} | acc {:.4f} | val_acc {:.4f} | test_acc {:.4f}".format(
                        epoch, loss.item(), np.mean(dur), train_acc, val_acc, test_acc))

                end_time = time.time()
                print("Each Epoch Cost Time: %f " % ((end_time - begin_time) / epoch))
        print(f"val_score:{model_val_acc},test_score:{best_performance}")
        if return_best:
            return model, model_val_acc, best_performance
        else:
            return model, model_val_acc
