import argparse
import sys

import torch
import torch.nn.functional as F

sys.path.extend(['../'])
from models.gnn_citation_manager import CitationGNN, evaluate, process_action, GraphNet


def build_args():
    parser = argparse.ArgumentParser(description='GraphNAS')
    parser.add_argument('--random_seed', type=int, default=123)
    parser.add_argument("--cuda", type=bool, default=True, required=False,
                        help="run in cuda mode")
    # child model
    parser.add_argument("--dataset", type=str, default="cora", required=False,
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


def build_model_optimizer(actions, with_param=False):
    actions, model, self = build_model(actions)
    if with_param:
        model.load_param(self.shared_params)
    model.cuda()
    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
    if with_param:
        self.load_optimizer(optimizer)

    try:
        model, val_acc = self.run_model(model, optimizer, self.loss_fn, self.data, 40,
                                        half_stop_score=max(self.reward_manager.get_top_average() * 0.7, 0.4))
    except RuntimeError as e:
        if "cuda" in str(e) or "CUDA" in str(e):
            print(e)
            val_acc = 0
        else:
            raise e
    reward = self.reward_manager.get_reward(val_acc)
    model.eval()
    eval_model(model, self.data)
    self.save_param(model, update_all=True)
    self.save_optimizer(optimizer)
    torch.save(model.state_dict(), "model/cora_state_dict.pkl")


def build_model(actions):
    args = build_args()
    self = CitationGNN(args)
    torch.cuda.empty_cache()
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    actions = process_action(actions, "two", self.args)
    model = GraphNet(actions, self.in_feats, self.n_classes, drop_out=self.args.in_drop, multi_label=False,
                     batch_normal=False)
    return actions, model, self


def eval(actions):
    actions, model, self = build_model(actions)
    model.cuda()

    eval_model(model, self.data)


def eval_model(model, data):
    features, g, labels, mask, val_mask, test_mask, n_edges = CitationGNN.prepare_data(data)
    # model.eval()
    logits = model(features, g)
    logits = F.log_softmax(logits, 1)
    train_acc = evaluate(logits, labels, mask)
    val_acc = evaluate(logits, labels, val_mask)
    test_acc = evaluate(logits, labels, test_mask)
    print("acc {:.4f} | val_acc {:.4f} | test_acc {:.4f}".format(train_acc, val_acc, test_acc))


def eval_actions(actions):
    args = build_args()
    self = CitationGNN(args)
    self.retrain(actions)


if __name__ == "__main__":
    actions = ['none', 'mlp', 'leaky_relu', 16, 64, 'bilinear', 'sum', 'elu', 4, 7]
    eval_actions(actions)


def test_param_save(actions):
    build_model_optimizer(actions, with_param=True)
    torch.cuda.empty_cache()
    actions, model, self = build_model(actions)
    model.load_param(self.shared_params)
    model.cuda()
    model.eval()
    eval_model(model, self.data)
