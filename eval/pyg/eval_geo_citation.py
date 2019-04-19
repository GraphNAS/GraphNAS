import argparse
import os.path as osp

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from models.geo.geo_gnn import GraphNet
# sys.path.extend(['/GraphNAS'])
from models.gnn_citation_manager import CitationGNN, process_action
from models.geo.geo_gnn_citation_manager import GeoCitationManager
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)


def build_args():
    parser = argparse.ArgumentParser(description='GraphNAS')
    parser.add_argument('--random_seed', type=int, default=123)
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
    parser.add_argument("--param_file", type=str, default="cora_test.pkl",
                        help="learning rate")
    parser.add_argument("--optim_file", type=str, default="opt_cora_test.pkl",
                        help="optimizer save path")
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--max_param', type=float, default=5E6)
    args = parser.parse_args()

    return args


def build_model(actions):
    args = build_args()
    self = CitationGNN(args)
    torch.cuda.empty_cache()

    actions = process_action(actions, "two", args)
    model = GraphNet(actions, self.in_feats, self.n_classes, drop_out=self.args.in_drop, multi_label=False,
                     batch_normal=False)
    return actions, model, self


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(data.x, data.edge_index)[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


def test(model, data):
    model.eval()
    logits, accs = model(data.x, data.edge_index), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


def evaluate_gnn():
    global actions
    actions, model, self = build_model(actions)
    dataset = 'Citeseer'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
    dataset = Planetoid(path, dataset, T.NormalizeFeatures())
    data = dataset[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    for epoch in range(1, 201):
        train(model, optimizer, data)
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, *test(model, data)))


if __name__ == "__main__":
    # actions = ['gat', 'sum', 'elu', 8, 8, 'gat', 'sum', 'elu', 1, 7]
    actions = ['gat_sym', 'sum', 'linear', 4, 64, 'generalized_linear', 'sum', 'linear', 4, 7]
    actions = ['generalized_linear', 'sum', 'tanh', 1, 256, 'cos', 'sum', 'elu', 4, 7]
    actions = ['cos', 'sum', 'linear', 4, 16, 'linear', 'sum', 'linear', 1, 7]
    actions = ['gat_sym', 'sum', 'tanh', 2, 64, 'linear', 'sum', 'elu', 1, 7]
    actions = ['const', 'sum', 'elu', 4, 8, 'generalized_linear', 'sum', 'linear', 4, 7]
    actions = ['linear', 'sum', 'elu', 4, 64, 'gat_sym', 'sum', 'elu', 8, 6]

    actions = ['const', 'sum', 'relu6', 2, 128, 'gat', 'sum', 'linear', 2, 7]
    args = build_args()
    manager = GeoCitationManager(args)
    for i in range(20):
        manager.train(actions)
