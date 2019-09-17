import argparse

import numpy as np
import torch


from graphnas_variants.micro_graphnas.micro_model_manager import MicroCitationManager

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
    parser.add_argument('--max_param', type=float, default=5E6)
    parser.add_argument('--supervised', type=bool, default=True)
    parser.add_argument('--layers_of_child_model', type=int, default=2)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = build_args()
    gnn_list = [
        {'action': [0, 'gat_6', 0, 'gcn', 0, 'gcn', 2, 'arma', 'tanh', 'concat'], 'hyper_param': [0.01, 0.9, 0.0001, 64]},
        {'action': [0, 'linear', 0, 'gat_6', 'linear', 'concat'], 'hyper_param': [0.005, 0.8, 1e-05, 128]},
        {'action': [1, 'gat_8', 0, 'arma', 'tanh', 'add'], 'hyper_param': [0.01, 0.4, 5e-05, 64]}
    ]

    dataset_list = ["Cora", "Citeseer", "Pubmed"]
    for shuffle in [False, True]:
        for dataset, actions in zip(dataset_list, gnn_list):
            args.dataset = dataset
            manager = MicroCitationManager(args)
            test_scores_list = []
            for i in range(100):
                if shuffle:
                    manager.shuffle_data()
                val_acc, test_acc = manager.evaluate(actions)
                test_scores_list.append(test_acc)
            print("_" * 80)
            test_scores_list.sort()
            if shuffle:
                print(dataset, "randomly split results:", np.mean(test_scores_list[5:-5]), np.std(test_scores_list[5:-5]))
            else:
                print(dataset, "fixed split results:", np.mean(test_scores_list[5:-5]), np.std(test_scores_list[5:-5]))

