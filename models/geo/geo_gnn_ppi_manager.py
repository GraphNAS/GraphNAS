import os.path as osp

import torch
from sklearn import metrics
from torch_geometric.data import DataLoader
from torch_geometric.datasets import PPI

from models.geo.geo_gnn import GraphNet
from models.gnn_manager import GNNManager
from models.model_utils import TopAverage, process_action


def load_data():
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PPI')
    train_dataset = PPI(path, split='train')
    val_dataset = PPI(path, split='val')
    test_dataset = PPI(path, split='test')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_loader, val_loader, test_loader


class GeoPPIManager(GNNManager):
    def __init__(self, args):
        super(GeoPPIManager, self).__init__(args)

        self.train_loader, self.val_loader, self.test_loader = load_data()
        self.device = torch.device('cuda' if args.cuda else 'cpu')
        # self.data.to(self.device)

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

    def build_gnn(self, actions):
        model = GraphNet(actions, self.in_feats, self.n_classes, drop_out=self.args.in_drop, multi_label=True,
                         batch_normal=False, residual=True)
        return model

    def load_param(self):
        if hasattr(self.args, "share_param"):
            if not self.args.share_param:  # 不共享参数
                return
        if osp.exists(self.param_file):
            self.shared_params = torch.load(self.param_file)

    def save_param(self, model, update_all=False):
        if hasattr(self.args, "share_param"):
            if not self.args.share_param:
                return
        model.cpu()
        if isinstance(model, GraphNet):
            self.shared_params = model.get_param_dict(self.shared_params, update_all)
        torch.save(self.shared_params, self.param_file)

    def train(self, actions, format="two"):

        actions = process_action(actions, format, self.args)
        model = self.build_gnn(actions)
        print("train action:", actions)

        # eval number of params of model, big model will be drop
        num_params = sum([param.nelement() for param in model.parameters()])
        if num_params > self.args.max_param:
            print(f"model too big, num_param more than {self.args.max_param}")
            del model
            return None

        # share params
        model.load_param(self.shared_params)

        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_op = torch.nn.BCEWithLogitsLoss()
        try:
            for epoch in range(1, self.epochs + 1):
                model.train()
                total_loss = self.run_model(model, optimizer, loss_op)
                f1_val = self.test(model, self.val_loader)
                f1_test = self.test(model, self.test_loader)
                print('Epoch: {:02d}, Loss: {:.4f}, val_acc: {:.4f}, test_acc:{:.4f}'.format(epoch, total_loss, f1_val,
                                                                                             f1_test))
        except RuntimeError as e:
            if "cuda" in str(e) or "CUDA" in str(e):
                print(e)
            else:
                raise e
        # run on val data
        f1_val = self.test(model, self.val_loader)
        reward = self.reward_manager.get_reward(f1_val)
        self.save_param(model, update_all=(reward > 0))  # 对模型有利，保留训练参数
        return reward, f1_val

    def run_model(self, model, optimizer, loss_op):
        model.train()
        total_loss = 0
        for data in self.train_loader:
            num_graphs = data.num_graphs
            data.batch = None
            data = data.to(self.device)
            optimizer.zero_grad()
            loss = loss_op(model(data.x, data.edge_index), data.y)
            total_loss += loss.item() * num_graphs
            loss.backward()
            optimizer.step()
        return total_loss / len(self.train_loader.dataset)

    def test(self, model, loader):
        model.eval()

        total_micro_f1 = 0
        for data in loader:
            torch.cuda.empty_cache()
            with torch.no_grad():
                out = model(data.x.to(self.device), data.edge_index.to(self.device))
            pred = (out > 0).float().cpu()
            micro_f1 = metrics.f1_score(data.y, pred, average='micro')
            total_micro_f1 += micro_f1 * data.num_graphs
        return total_micro_f1 / len(loader.dataset)

    # evaluate model from scratch
    def retrain(self, actions, format="two"):
        return self.train(actions, format)

    # evaluate model with a few training or no training
    def test_with_param(self, actions, format="two", with_retrain=False):
        try:
            return self.train(actions, format)
        except RuntimeError as e:
            if "CUDA" in str(e):
                return None
            else:
                raise e
