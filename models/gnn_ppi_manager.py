import argparse
import json
import os
import time
from math import ceil

import networkx as nx
import numpy as np
import torch
from dgl import DGLGraph
from networkx.readwrite import json_graph

from models.gnn import GraphNet
from models.gnn_manager import GNNManager
from models.model_utils import EarlyStop, TopAverage
from models.model_utils import process_action, calc_f1


def build_args():
    parser = argparse.ArgumentParser(description='PPI')
    parser.add_argument('--random_seed', type=int, default=123)
    parser.add_argument("--cuda", type=bool, default=True, required=False,
                        help="run in cuda mode")
    # register_data_args(parser)
    parser.add_argument("--epochs", type=int, default=2,
                        help="number of training epochs")
    parser.add_argument("--num_train_graph", type=int, default=10,
                        help="number of graphs in each training epochs")
    parser.add_argument("--retrain_epochs", type=int, default=200,
                        help="number of retrain epochs")
    parser.add_argument("--in-feats", type=int, default=50,
                        help="number of input features")
    parser.add_argument("--num-class", type=int, default=121,
                        help="number of output units")
    parser.add_argument("--multi_label", type=bool, default=True,
                        help="multi_label or single_label task")
    parser.add_argument("--in-drop", type=float, default=0,
                        help="input feature dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument("--param_file", type=str, default="test_share.pkl",
                        help=" shared parameters save path")
    parser.add_argument("--optim_file", type=str, default="opt_test_share.pkl",
                        help="optimizer save path")
    parser.add_argument("--retrain_filename", type=str, default="retrain_model.pkl",
                        help="final model save path")
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--max_param', type=float, default=5E6)
    args = parser.parse_args()

    return args


class PPIGNN(GNNManager):

    def __init__(self, args):
        super(PPIGNN, self).__init__(args)

        self.group_feats, self.group_edge, self.group_labels, self.group_graphs = load_data("new_ppi.npy")
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

        self.load_param()

    def load_param(self):
        if hasattr(self.args, "share_param"):
            if not self.args.share_param:  # 不共享参数
                return
        if os.path.exists(self.param_file):
            self.shared_params = torch.load(self.param_file)

    @staticmethod
    def _merge_optimizer(new_state_dict, shared_state_dict, merge=False):
        for param in shared_state_dict['state']:
            if param in new_state_dict['param_groups'][0]['params']:
                new_state_dict['state'][param] = shared_state_dict['state'][param]
            else:
                if merge:
                    new_state_dict['state'][param] = shared_state_dict['state'][param]
                    new_state_dict['param_groups'][0]['params'].append(param)
        return new_state_dict

    def save_param(self, model, update_all=False):
        if hasattr(self.args, "share_param"):
            if not self.args.share_param:
                return
        model.cpu()
        if isinstance(model, GraphNet):
            self.shared_params = model.get_param_dict(self.shared_params, update_all)
        torch.save(self.shared_params, self.param_file)

    def train(self, actions=None, dataset="ppi", format="two"):
        torch.manual_seed(self.args.random_seed)
        if self.args.cuda:
            torch.cuda.empty_cache()
            torch.cuda.manual_seed(self.args.random_seed)

        model = self.build_model(actions, format)
        print("train action:", actions)

        # eval number of params of model, big model will be drop
        num_params = sum([param.nelement() for param in model.parameters()])
        if num_params > self.args.max_param:
            print(f"model too big, num_param more than {self.args.max_param}")
            del model
            return None

        # share params
        model.load_param(self.shared_params)
        if self.args.cuda:
            model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for epoch in range(1, self.epochs + 1):
            train_loss = []
            train_score = []

            for i in range(10):
                try:
                    features, g, labels, n_edges = prepare_data(self.group_feats, self.group_graphs,
                                                                self.group_labels, i, cuda=self.args.cuda)
                    model.train()
                    self.run_model(epoch, features, g, i, labels, model, n_edges, optimizer,
                                   train_loss, train_score)
                except RuntimeError as e:
                    if "cuda" in str(e) or "CUDA" in str(e):
                        print(e)
                    else:
                        raise e

                try:
                    self.clear_graph(g)  # release memory used by dgl graph
                    del features, g, labels, n_edges
                except Exception as e:
                    pass

            if epoch == ceil(self.epochs / 2):  # early stop
                if self.early_stop_manager.should_stop(np.mean(train_loss), np.mean(train_score), 0, 0):
                    break
        # run on val data
        loss_val, f1_val = self.test(model)
        reward = self.reward_manager.get_reward(f1_val)
        self.save_param(model, update_all=(reward > 0))  # 对模型有利，保留训练参数
        torch.cuda.empty_cache()
        return reward, f1_val

    def build_model(self, actions, format):
        actions = process_action(actions, format, self.args)
        model = GraphNet(actions, self.in_feats, self.n_classes, drop_out=self.drop_out,
                         multi_label=self.multi_label)
        return model

    def run_model(self, epoch, features, g, i, labels, model, n_edges, optimizer, train_loss, train_score):
        t0 = time.time()
        # forward
        logits = model(features, g)
        logp = torch.sigmoid(logits)
        f1_train = calc_f1(logp, labels)
        loss = self.loss_fn(logp, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(float(loss))
        train_score.append(f1_train)
        print("Epoch {:05d} | Graph {:05d} | Loss {:.4f} | Time(s) {:.4f} | f1 {:.4f}, {:.4f}".format(
            epoch, i, loss.item(), time.time() - t0, f1_train[0], f1_train[1]))

    def retrain(self, actions, format="two", force=False, with_param=False):
        torch.manual_seed(self.args.random_seed)
        if self.args.cuda:
            torch.cuda.empty_cache()
            torch.cuda.manual_seed(self.args.random_seed)

        # create model
        model = self.build_model(actions, format)
        print("retrain action: ", actions)
        # check model
        num_params = sum([param.nelement() for param in model.parameters()])
        print(num_params)
        if num_params > self.args.max_param and not force:
            print(f"model too big, num_param more than {self.args.max_param}")
            del model
            return None

        if with_param:
            model.load_param(self.shared_params)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.load_optimizer(optimizer)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.args.cuda:
            model.cuda()
        # use optimizer

        begin_time = time.time()

        for epoch in range(1, self.retrain_epochs + 1):
            train_loss = []
            train_score = []
            for i in range(10):
                try:
                    features, g, labels, n_edges = prepare_data(self.group_feats, self.group_graphs, self.group_labels,
                                                                i, cuda=self.args.cuda)
                    model.train()

                    self.run_model(epoch, features, g, i, labels, model, n_edges, optimizer, train_loss,
                                   train_score)

                    end_time = time.time()
                    print("Each Epoch Cost Time: %f " % ((end_time - begin_time) / epoch))

                except RuntimeError as e:
                    if "cuda" in str(e) or "CUDA" in str(e):
                        print(e)
                    else:
                        raise e
                try:
                    self.clear_graph(g)
                    del features, g, labels, n_edges
                except Exception as e:
                    pass
                torch.cuda.empty_cache()
            self.test(model)
            self.test(model, 11)
        model_path = self.args.retrain_filename
        opt_path = self.args.optim_file
        if model_path:
            # torch.save(model.state_dict(), model_path)
            self.save_param(model, update_all=True)
            torch.save(optimizer.state_dict(), opt_path)
        del model
        torch.cuda.empty_cache()

    def test(self, model, graph_index=10):
        torch.cuda.empty_cache()
        model.eval()
        try:
            loss = []
            f1_score = []
            i = graph_index
            features, g, labels, n_edges = prepare_data(self.group_feats, self.group_graphs, self.group_labels,
                                                        graph_index, cuda=self.args.cuda)
            output = model(features, g)
            output = torch.sigmoid(output)
            f1_test = calc_f1(output, labels)
            f1_score.append(f1_test[0])
            loss_value = float(self.loss_fn(output, labels))
            loss.append(loss_value)
            print('Graph: {:04d}'.format(i + 1),
                  'loss:{:04f}'.format(loss_value),
                  'f1_test: {:04f}, {:04f}'.format(f1_test[0], f1_test[1]))
        # val loss, val f1, test loss, test score
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(e)
            else:
                raise e
            loss.append(2 ** 10)
            f1_score.append(0)
            torch.cuda.empty_cache()

        try:
            self.clear_graph(g)
            del features, g, labels, n_edges
        except Exception as e:
            pass

        torch.cuda.empty_cache()
        return loss[0], f1_score[0]

    @staticmethod
    def clear_graph(g):
        keys = list(g.ndata.keys())
        for each in keys:
            g.pop_n_repr(each)
        keys = list(g.edata.keys())
        for each in keys:
            g.pop_n_repr(each)

    def test_with_param(self, actions=None, format="two", with_retrain=True):
        torch.cuda.empty_cache()
        if with_retrain:
            self.train(actions)  # retrain

        model = self.build_model(actions, format)
        print("evaluate action:", actions)

        num_params = sum([param.nelement() for param in model.parameters()])
        if num_params > self.args.max_param:
            print("model too big, num_param more than 10M")
            del model
            return None
        if self.args.cuda:
            model.cuda()

        model.load_param(self.shared_params)
        # use optimizer
        try:
            val_loss, val_f1 = self.test(model)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(e)
            else:
                raise e
            val_loss, val_f1 = 0, 0
        del model
        torch.cuda.empty_cache()
        return val_loss, val_f1


def prepare_data(group_feats, group_graphs, group_labels, i, cuda=True):
    features = torch.FloatTensor(group_feats[i])
    labels = torch.FloatTensor(group_labels[i])
    g = group_graphs[i]
    n_edges = g.number_of_edges()

    # torch.cuda.set_device(0)
    # create DGL graph
    g = DGLGraph(g)
    # add self loop
    g.add_edges(g.nodes(), g.nodes())
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0

    if cuda:
        # torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)
    return features, g, labels, n_edges


base = os.path.split(os.path.realpath(__file__))[0]


def load_data(save_file="ppi_data.npy"):
    if os.path.exists(save_file):
        return np.load(save_file)
    global base
    base += "/../"
    filepath = base + "/p2p_dataset/ppi-G.json"
    graph_data = json_graph.node_link_graph(json.load(open(filepath)))
    feats = np.load(base + "/p2p_dataset/ppi-feats.npy")
    labels = json.load(open(base + "/p2p_dataset/ppi-class_map.json"))

    feats = standarizing_features(graph_data, feats)

    components = list(nx.connected_components(graph_data))

    all_data = []
    for each in components:
        if len(each) > 2:
            all_data.append(each)

    lengths = [len(each) for each in all_data[:20]]
    index = np.argsort(lengths)
    new_data = []
    for i in range(10):
        new_data.append(list(all_data[i]))
        new_data[-1].extend(all_data[19 - i])
    for i in range(20, 24, 2):
        new_data.append(list(all_data[i]))
        new_data[-1].extend(all_data[i + 1])
    all_data = new_data
    max_subgraph_length = len(max(all_data, key=len))
    id_map = {}
    for i, comp in enumerate(all_data):
        for j, id_ in enumerate(comp):
            id_map[id_] = [i, j]

    group_feats = build_feats(all_data, feats)
    group_labels = build_labels(all_data, labels)
    group_edge = build_edge_index(all_data, graph_data, id_map)
    group_graphs = build_sub_graph(group_edge)
    np.save(save_file, [group_feats, group_edge, group_labels, group_graphs])
    return group_feats, group_edge, group_labels, group_graphs


def build_sub_graph(group_edge):
    sub_graphs = []
    for edge_list in group_edge:
        g = nx.Graph()
        g.add_edges_from(edge_list.T)
        sub_graphs.append(g)
    return sub_graphs


def standarizing_features(G, features_):
    from sklearn.preprocessing import StandardScaler
    train_ids = np.array([n for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
    train_feats = features_[train_ids]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    features_ = scaler.transform(features_)
    # features = sp.csr_matrix(features_).tolil()

    return features_


def build_edge_index(all_data, graph_data, id_map):
    group_edges = []
    for comp in all_data:
        left = []
        right = []
        for start in comp:
            i = id_map[start][-1]
            left.append(i)
            right.append(i)
            for end in graph_data[start]:
                j = id_map[end][-1]
                left.append(i)
                right.append(j)
        group_edges.append(np.array([left, right]))
    return group_edges


def build_feats(all_data, all_feats):
    group_feats = []
    for comp in all_data:
        tmp = []
        for id_ in comp:
            tmp.append(all_feats[id_])
        group_feats.append(tmp)
    return np.array(group_feats)


def build_labels(all_data, labels):
    group_labels = []
    for comp in all_data:
        tmp = []
        for id_ in comp:
            tmp.append(labels[str(id_)])
        group_labels.append(tmp)
    return np.array(group_labels)
