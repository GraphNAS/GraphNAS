import argparse
import torch
import torch.nn.functional as F
from torch_geometric.nn import ARMAConv

from baseline import get_planetoid_dataset, random_planetoid_splits, run

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="Pubmed")
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=100)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--early_stopping', type=int, default=100)
parser.add_argument('--hidden', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--num_stacks', type=int, default=2)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--shared_weights', type=bool, default=False)
parser.add_argument('--skip_dropout', type=float, default=0.75)
args = parser.parse_args()


class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.conv1 = ARMAConv(
            dataset.num_features,
            args.hidden,
            args.num_stacks,
            args.num_layers,
            args.shared_weights,
            dropout=args.skip_dropout)
        self.conv2 = ARMAConv(
            args.hidden,
            dataset.num_classes,
            args.num_stacks,
            args.num_layers,
            args.shared_weights,
            dropout=args.skip_dropout)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

if __name__ == "__main__":
    dataset = get_planetoid_dataset(args.dataset, args.normalize_features)
    permute_masks = random_planetoid_splits if args.random_splits else None
    run(dataset, Net(dataset), args.runs, args.epochs, args.lr, args.weight_decay,
        args.early_stopping, permute_masks)
