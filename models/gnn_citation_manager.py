"""
Graph Attention Networks
Paper: https://arxiv.org/abs/1710.10903
Code: https://github.com/PetarV-/GAT
GAT with batch processing
"""

import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import load_data
from models.gnn import GraphNet
from models.model_utils import EarlyStop, TopAverage, process_action


def load(args, save_file="citation.npy"):
    if os.path.exists(save_file):
        return np.load(save_file).tolist()
    else:
        datas = load_data(args)
        np.save(save_file, datas)
        return datas


def evaluate(output, labels, mask):
    # with torch.no_grad():
    _, indices = torch.max(output, dim=1)
    correct = torch.sum(indices[mask] == labels[mask])
    return correct.item() * 1.0 / mask.sum().item()


class CitationGNN(object):

    def __init__(self, args, data_function=None):
        self.data = load(args)
        self.args = args
        if hasattr(args, 'dataset'):

            self.args.in_feats = self.in_feats = self.data.features.shape[1]
            self.args.num_class = self.n_classes = self.data.num_labels

        else:
            raise Exception("args has no dataset")
        self.early_stop_manager = EarlyStop(10)
        self.reward_manager = TopAverage(10)

        self.args = args
        self.in_feats = args.in_feats
        self.n_classes = args.num_class
        self.drop_out = args.in_drop
        self.multi_label = args.multi_label
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.retrain_epochs = args.retrain_epochs
        self.loss_fn = torch.nn.BCELoss()
        self.epochs = args.epochs
        self.train_graph_index = 0
        self.train_set_length = 10

        self.param_file = args.param_file
        self.shared_params = None

        self.loss_fn = torch.nn.functional.nll_loss

    def load_param(self):
        if hasattr(self.args, "share_param"):
            if not self.args.share_param:  # 不共享参数
                return
        if os.path.exists(self.param_file):
            self.shared_params = torch.load(self.param_file)

    def save_param(self, model, update_all=False):
        if hasattr(self.args, "share_param"):
            if not self.args.share_param:
                return
        model.cpu()
        if isinstance(model, GraphNet):
            self.shared_params = model.get_param_dict(self.shared_params, update_all)
        torch.save(self.shared_params, self.param_file)

    def train(self, actions=None, dataset="cora", format="two"):
        actions = process_action(actions, format, self.args)
        print("train action:", actions)

        # create model
        model = self.build_gnn(actions)

        # share params
        # model.load_param(self.shared_params)
        if self.args.cuda:
            model.cuda()

        # use optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        try:
            model, val_acc = self.run_model(model, optimizer, self.loss_fn, self.data, self.epochs, cuda=self.args.cuda,
                                            half_stop_score=max(self.reward_manager.get_top_average() * 0.7, 0.4))
        except RuntimeError as e:
            if "cuda" in str(e) or "CUDA" in str(e):
                print(e)
                val_acc = 0
            else:
                raise e
        reward = self.reward_manager.get_reward(val_acc)
        self.save_param(model, update_all=(reward > 0))

        with open('citetion_result.txt', "a") as file:
            file.write(str(actions))

            file.write(",")
            file.write(str(reward))

            file.write(",")
            file.write(str(val_acc))
            file.write("\n")

        return reward, val_acc

    def build_gnn(self, actions):
        model = GraphNet(actions, self.in_feats, self.n_classes, drop_out=self.args.in_drop, multi_label=False,
                         batch_normal=False)
        return model

    def evaluate(self, actions=None, dataset="cora", format="two"):
        if self.args.cuda:
            torch.cuda.empty_cache()
        # without random seed
        # torch.manual_seed(123)
        # torch.cuda.manual_seed_all(123)
        actions = process_action(actions, format, self.args)
        print("train action:", actions)

        # create model
        model = self.build_gnn(actions)
        if self.args.cuda:
            model.cuda()
        total = sum([param.nelement() for param in model.parameters()])
        print(total)
        # use optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        try:
            model, val_acc, best_test_acc = self.run_model(model, optimizer, self.loss_fn, self.data, self.epochs,
                                                           half_stop_score=max(
                                                               self.reward_manager.get_top_average() * 0.7, 0.4),
                                                           return_best=True, cuda=self.args.cuda)
        except RuntimeError as e:
            if "cuda" in str(e):
                print(e)
                val_acc = 0
            else:
                raise e

        return best_test_acc

    def retrain(self, actions, format="two"):
        torch.manual_seed(self.args.random_seed)
        if self.args.cuda:
            torch.cuda.empty_cache()
            torch.cuda.manual_seed_all(self.args.random_seed)
        actions = process_action(actions, format, self.args)
        print("retrain action:", actions)

        # create model
        model = self.build_gnn(actions)
        if self.args.cuda:
            model.cuda()

        # use optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        # different from train
        try:
            model, val_acc = self.run_model(model, optimizer, self.loss_fn, self.data, self.epochs, cuda=self.args.cuda,
                                            half_stop_score=max(self.reward_manager.get_top_average() * 0.7, 0.4))
        except RuntimeError as e:
            if "cuda" in str(e):
                print(e)
                val_acc = 0
            else:
                raise e
        reward = self.reward_manager.get_reward(val_acc)
        self.save_param(model, update_all=(reward > 0))
        return reward, val_acc

    def test_with_param(self, actions=None, dataset="cora", format="two", with_retrain=True):
        return self.train(actions, dataset, format)

    @staticmethod
    def run_model(model, optimizer, loss_fn, data, epochs, early_stop=5, tmp_model_file="citation_testing_2.pkl",
                  half_stop_score=0, return_best=False, cuda=True, need_early_stop=False):

        early_stop_manager = EarlyStop(early_stop)
        # initialize graph
        dur = []
        begin_time = time.time()
        features, g, labels, mask, val_mask, test_mask, n_edges = CitationGNN.prepare_data(data, cuda)
        saved = False
        best_performance = 0
        for epoch in range(1, epochs + 1):
            should_break = False
            # i = self.train_graph_index % self.train_set_length

            model.train()

            t0 = time.time()
            # forward
            logits = model(features, g)
            logits = F.log_softmax(logits, 1)
            loss = loss_fn(logits[mask], labels[mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.eval()
            logits = model(features, g)
            logits = F.log_softmax(logits, 1)
            train_acc = evaluate(logits, labels, mask)
            train_loss = float(loss)
            dur.append(time.time() - t0)

            val_loss = float(loss_fn(logits[val_mask], labels[val_mask]))
            val_acc = evaluate(logits, labels, val_mask)
            test_acc = evaluate(logits, labels, test_mask)
            if test_acc > best_performance:
                best_performance = test_acc
            print(
                "Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f} | acc {:.4f} | val_acc {:.4f} | test_acc {:.4f}".format(
                    epoch, loss.item(), np.mean(dur), train_acc, val_acc, test_acc))

            end_time = time.time()
            print("Each Epoch Cost Time: %f " % ((end_time - begin_time) / epoch))
            # print("Test Accuracy {:.4f}".format(acc))
            if early_stop_manager.should_save(train_loss, train_acc, val_loss, val_acc):
                saved = True
                torch.save(model.state_dict(), tmp_model_file)
            if need_early_stop and early_stop_manager.should_stop(train_loss, train_acc, val_loss, val_acc):
                should_break = True
            if should_break and epoch > 50:
                print("early stop")
                break
            if half_stop_score > 0 and epoch > (epochs / 2) and val_acc < half_stop_score:
                print("half_stop")
                break
        if saved:
            model.load_state_dict(torch.load(tmp_model_file))
        model.eval()
        val_acc = evaluate(model(features, g), labels, val_mask)
        print(evaluate(model(features, g), labels, test_mask))
        if return_best:
            return model, val_acc, best_performance
        else:
            return model, val_acc

    @staticmethod
    def prepare_data(data, cuda=True):
        features = torch.FloatTensor(data.features)
        labels = torch.LongTensor(data.labels)
        mask = torch.ByteTensor(data.train_mask)
        test_mask = torch.ByteTensor(data.test_mask)
        val_mask = torch.ByteTensor(data.val_mask)
        n_edges = data.graph.number_of_edges()
        # create DGL graph
        g = DGLGraph(data.graph)
        # add self loop
        g.add_edges(g.nodes(), g.nodes())
        degs = g.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0

        if cuda:
            features = features.cuda()
            labels = labels.cuda()
            norm = norm.cuda()
        g.ndata['norm'] = norm.unsqueeze(1)
        return features, g, labels, mask, val_mask, test_mask, n_edges
