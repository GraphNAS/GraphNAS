import argparse
import os.path as osp
import time

import torch
from sklearn import metrics
from torch_geometric.data import DataLoader
from torch_geometric.datasets import PPI

from models.geo.geo_gnn import GraphNet
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
    parser.add_argument("--dataset", type=str, default="cora", required=False,
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
    parser.add_argument("--param_file", type=str, default="cora_test.pkl",
                        help="learning rate")
    parser.add_argument("--optim_file", type=str, default="opt_cora_test.pkl",
                        help="optimizer save path")
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--max_param', type=float, default=5E6)
    args = parser.parse_args()

    return args


def build_model(actions):
    args = build_args()

    model = GraphNet(actions, 50, 121, drop_out=0, multi_label=True,
                     batch_normal=False, residual=True)
    return model


def train(model, optimizer, train_loader, device, loss_op):
    model.train()
    total_loss = 0
    for data in train_loader:
        torch.cuda.empty_cache()
        num_graphs = data.num_graphs
        data.batch = None
        data = data.to(device)
        optimizer.zero_grad()
        loss = loss_op(model(data.x, data.edge_index), data.y)
        total_loss += loss.item() * num_graphs
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader.dataset)


def test(model, loader, device):
    model.eval()

    total_micro_f1 = 0
    for data in loader:
        with torch.no_grad():
            out = model(data.x.to(device), data.edge_index.to(device))
        pred = (out > 0).float().cpu()
        micro_f1 = metrics.f1_score(data.y, pred, average='micro')
        total_micro_f1 += micro_f1 * data.num_graphs
    return total_micro_f1 / len(loader.dataset)


def evaluate_gnn():
    global actions
    model = build_model(actions)

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PPI')
    train_dataset = PPI(path, split='train')
    val_dataset = PPI(path, split='val')
    test_dataset = PPI(path, split='test')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    loss_op = torch.nn.BCEWithLogitsLoss()

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(1, 201):
        loss = train(model, optimizer, train_loader, device, loss_op)
        acc = test(model, val_loader, device)
        test_acc = test(model, test_loader, device)
        print('Epoch: {:02d}, Loss: {:.4f}, val_acc: {:.4f}, test_acc:{:.4f}'.format(epoch, loss, acc, test_acc))


def evaluate_gnn_share_param(actions, gpu=1, epochs=100, load_param=True, save_param=True):
    print(actions)
    model = build_model(actions)

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PPI')
    train_dataset = PPI(path, split='train')
    val_dataset = PPI(path, split='val')
    test_dataset = PPI(path, split='test')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    loss_op = torch.nn.BCEWithLogitsLoss()

    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    if load_param:
        model.load_param(torch.load("ppi_gnn"))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    start_time = time.time()
    for epoch in range(1, epochs+1):

        loss = train(model, optimizer, train_loader, device, loss_op)
        acc = test(model, val_loader, device)
        test_acc = test(model, test_loader, device)
        print('Epoch: {:02d}, Loss: {:.4f}, val_acc: {:.4f}, test_acc:{:.4f}'.format(epoch, loss, acc, test_acc))
    end_time = time.time()
    print(end_time - start_time)
    if save_param:
        state_dict = model.get_param_dict()
        torch.save(state_dict, "ppi_gnn")


def test_manager():
    args = build_args()
    manager = GeoPPIManager(args)
    manager.train(actions)


if __name__ == "__main__":
    actions = ['gat', 'sum', 'elu', 4, 256, 'gat', 'sum', 'elu', 4, 256, 'gat', 'sum', 'linear', 6, 121]
    # actions = ['cos', 'sum', 'elu', 4, 256, 'cos', 'sum', 'elu', 4, 256, 'cos', 'sum', 'linear', 6, 121]
    # actions = ['gcn', 'mlp', 'tanh', 4, 32, 'gcn', 'mean', 'leaky_relu', 2, 64, 'gat_sym', 'mlp', 'elu', 1, 121]
    # evaluate_gnn()
    # test_manager()

    actions = ['const', 'mlp', 'elu', 16, 64, 'cos', 'mlp', 'linear', 8, 64, 'generalized_linear', 'max', 'linear', 4,
               121]
    # actions_2 = ['const', 'mean', 'tanh', 8, 128, 'gat', 'mlp', 'elu', 8, 32, 'gat', 'sum', 'linear', 6, 121]
    actions = ['linear', 'sum', 'relu', 16, 32,
               'linear', 'max', 'sigmoid', 6, 256,
               'gcn', 'max', 'linear', 1, 121]
    print("add skip_connection on layer 1")
    # graphnas
    actions = ['cos', 'max', 'tanh', 6, 64, 'gat_sym', 'mean', 'relu', 6, 128, 'gat', 'mlp', 'linear', 16, 121]
    # actions = ['linear', 'mlp', 'softplus', 16, 64, 'generalized_linear', 'sum', 'leaky_relu', 16, 32,
    #            'generalized_linear', 'sum', 'tanh', 16, 121]
    evaluate_gnn_share_param(actions, gpu=5, epochs=150, load_param=False, save_param=False)

    # random_action = ['cos', 'mean', 'elu', 16, 16, 'const', 'sum', 'tanh', 16, 128, 'gat', 'max', 'linear', 6, 121]
    # evaluate_gnn_share_param(random_action, gpu=3, epochs=150, load_param=False, save_param=False)

