import argparse
import os.path as osp

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from models.geo.geo_gnn import GraphNet
# sys.path.extend(['/GraphNAS'])
from models.gnn_citation_manager import CitationGNNManager, process_action
from models.geo.geo_gnn_citation_manager import GeoCitationManagerManager
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)


def build_args():
    parser = argparse.ArgumentParser(description='GraphNAS')
    parser.add_argument('--random_seed', type=int, default=123)
    parser.add_argument("--cuda", type=bool, default=True, required=False,
                        help="run in cuda mode")
    # child model
    parser.add_argument("--dataset", type=str, default="Pubmed", required=False,
                        help="The input dataset.")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--retrain_epochs", type=int, default=300,
                        help="number of training epochs")
    parser.add_argument("--multi_label", type=bool, default=False,
                        help="multi_label or single_label task")
    parser.add_argument("--residual", action="store_false",
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0.6,
                        help="input feature dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    # parser.add_argument("--lr", type=float, default=0.01,
    #                     help="learning rate")
    parser.add_argument("--param_file", type=str, default="cora_test.pkl",
                        help="learning rate")
    parser.add_argument("--optim_file", type=str, default="opt_cora_test.pkl",
                        help="optimizer save path")
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    # parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--max_param', type=float, default=5E6)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    actions = ['gat', 'sum', 'elu', 8, 8, 'gat', 'sum', 'elu', 1, 7]
    actions = ['cos', 'sum', 'tanh', 6, 64, 'linear', 'sum', 'leaky_relu', 1, 6]
    actions = ['generalized_linear', 'sum', 'tanh', 4, 32, 'cos', 'sum', 'elu', 1, 6]
    actions = ['const', 'sum', 'linear', 4, 32, 'linear', 'sum', 'tanh', 4, 3]
    args = build_args()
    manager = GeoCitationManagerManager(args)
    for i in range(20):
        manager.train(actions)
