import argparse

import torch

from models.geo.geo_gnn_ppi_manager import GeoPPIManager

# sys.path.extend(['/GraphNAS'])

torch.manual_seed(123)
torch.cuda.manual_seed_all(123)


def build_args():
    parser = argparse.ArgumentParser(description='GraphNAS')
    parser.add_argument('--random_seed', type=int, default=123)
    parser.add_argument("--cuda", type=bool, default=True, required=False,
                        help="run in cuda mode")
    # child model
    parser.add_argument("--in-feats", type=int, default=50,
                        help="number of input features")
    parser.add_argument("--num-class", type=int, default=121,
                        help="number of output units")
    parser.add_argument("--dataset", type=str, default="ppi", required=False,
                        help="The input dataset.")
    parser.add_argument("--epochs", type=int, default=50,
                        help="number of training epochs")
    parser.add_argument("--retrain_epochs", type=int, default=300,
                        help="number of training epochs")
    parser.add_argument("--multi_label", type=bool, default=False,
                        help="multi_label or single_label task")
    parser.add_argument("--residual", action="store_false",
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0,
                        help="input feature dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument("--param_file", type=str, default="ppi.pkl",
                        help="learning rate")
    parser.add_argument("--optim_file", type=str, default="ppi_optim.pkl",
                        help="optimizer save path")
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--max_param', type=float, default=5E6)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    actions = ['gat', 'sum', 'elu', 4, 256, 'gat', 'sum', 'elu', 4, 256, 'gat', 'sum', 'linear', 6, 121]
    args = build_args()
    manager = GeoPPIManager(args)
    manager.train(actions)
