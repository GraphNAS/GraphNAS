import argparse
import time

import torch

# sys.path.extend(['/GraphNAS'])
from models.gnn_citation_manager import CitationGNNManager


def build_args():
    parser = argparse.ArgumentParser(description='GraphNAS')
    parser.add_argument('--random_seed', type=int, default=123)
    parser.add_argument("--cuda", type=bool, default=True, required=False,
                        help="run in cuda mode")
    # child model
    parser.add_argument("--dataset", type=str, default="cora", required=False,
                        help="The input dataset.")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--retrain_epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--multi_label", type=bool, default=False,
                        help="multi_label or single_label task")
    parser.add_argument("--residual", action="store_false",
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0.6,
                        help="input feature dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument("--param_file", type=str, default="cora_test.pkl",
                        help="learning rate")
    parser.add_argument("--optim_file", type=str, default="opt_cora_test.pkl",
                        help="optimizer save path")
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--max_param', type=float, default=5E6)
    args = parser.parse_args()

    return args


def eval_actions(actions, run_random=False):
    args = build_args()
    if run_random:
        args.random_seed = time.time()

    self = CitationGNNManager(args)
    self.train(actions)


if __name__ == "__main__":

    # actions   ATT     ATT     ACT    K  DIM  ATT     ATT     ACT     K  DIM
    actions = ['none', 'sum', 'relu6', 2, 128, 'gat', 'sum', 'linear', 2, 7]
    for i in range(20):
        eval_actions(actions, run_random=True)

