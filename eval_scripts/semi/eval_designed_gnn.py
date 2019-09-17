import argparse

import numpy as np
import torch

from graphnas.gnn_model_manager import CitationGNNManager
from graphnas_variants.macro_graphnas.pyg.pyg_gnn_model_manager import GeoCitationManager

torch.manual_seed(123)
torch.cuda.manual_seed_all(123)


def build_args():
    parser = argparse.ArgumentParser(description='GraphNAS')
    parser.add_argument('--random_seed', type=int, default=123)
    parser.add_argument('--search_mode', type=str, default="nas")
    parser.add_argument('--format', type=str, default="two")
    parser.add_argument("--cuda", type=bool, default=True, required=False,
                        help="run in cuda mode")
    # child model
    parser.add_argument("--dataset", type=str, default="Cora", required=False,
                        help="The input dataset.")
    parser.add_argument("--epochs", type=int, default=300,
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
    args = build_args()

    gnn_list = [
        ['gat', 'sum', 'linear', 4, 128, 'linear', 'sum', 'elu', 8, 6],
        ['gcn', 'sum', 'tanh', 6, 64, 'cos', 'sum', 'tanh', 6, 3],
        ['const', 'sum', 'relu6', 2, 128, 'gat', 'sum', 'linear', 2, 7],
    ]
    dataset_list = ["Citeseer", "Pubmed", "cora"]
    base_list = ["pyg", "pyg", "dgl", ]
    for dataset, actions, base in zip(dataset_list, gnn_list, base_list):
        # if dataset == "cora":
        #     continue
        args.dataset = dataset
        if base == "dgl":
            manager = CitationGNNManager(args)
        else:
            manager = GeoCitationManager(args)
        test_scores_list = []
        for i in range(100):
            val_acc, test_acc = manager.evaluate(actions)
            test_scores_list.append(test_acc)
        print("_" * 80)
        test_scores_list.sort()
        print(dataset, np.mean(test_scores_list[5:-5]), np.std(test_scores_list[5:-5]))
