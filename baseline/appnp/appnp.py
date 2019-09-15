import argparse

import torch
import torch.nn.functional as F
from torch.nn import Linear

from baseline import get_planetoid_dataset, random_planetoid_splits, run
from torch_geometric.nn import APPNP

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="Pubmed", type=str)
parser.add_argument('--random_splits', type=bool, default=True)
parser.add_argument('--runs', type=int, default=100)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--early_stopping', type=int, default=10)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--alpha', type=float, default=0.1)
args = parser.parse_args()


class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.prop1 = APPNP(args.K, args.alpha)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)


def fix_size_split(data, train_num=1000, val_num=500, test_num=500):
    if hasattr(args, "train_num"):
        train_num = train_num
    assert train_num + val_num + test_num <= data.x.size(0), "not enough data"
    indices = torch.randperm(data.x.size(0))

    data.train_mask = index_to_mask(indices[:train_num], size=data.num_nodes)
    data.val_mask = index_to_mask(indices[train_num:train_num + val_num], size=data.num_nodes)
    data.test_mask = index_to_mask(indices[-test_num:], size=data.num_nodes)

    return data


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.uint8, device=index.device)
    mask[index] = 1
    return mask


if __name__ == "__main__":
    dataset = get_planetoid_dataset(args.dataset, args.normalize_features)
    # permute_masks = random_planetoid_splits if args.random_splits else None
    run(dataset, Net(dataset), args.runs, args.epochs, args.lr, args.weight_decay,
        args.early_stopping, None)
