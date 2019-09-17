import numpy as np
import torch
from eval_scripts.sup.eval_designed_gnn import build_args
from graphnas_variants.simple_graphnas.simple_model_manager import SimpleCitationManager

torch.manual_seed(123)
torch.cuda.manual_seed_all(123)


if __name__ == "__main__":
    args = build_args()
    args.supervised = True
    gnn_list = [
        {0: {'conv_type': 'gat_1', 'out_dim': 256}, 1: {'conv_type': 'sage', 'out_dim': 7}},
        {0: {'conv_type': 'arma', 'out_dim': 32}, 1: {'conv_type': 'arma', 'out_dim': 6}},
        {0: {'conv_type': 'sage', 'out_dim': 64}, 1: {'conv_type': 'arma', 'out_dim': 3}},
    ]
    dataset_list =["Cora", "Citeseer", "Pubmed"] #
    for dataset, actions in zip(dataset_list, gnn_list):
        args.dataset = dataset
        manager = SimpleCitationManager(args)
        test_scores_list = []
        for i in range(100):
            val_acc, test_acc = manager.evaluate(actions, format="simple")
            test_scores_list.append(test_acc)
        print("_" * 80)
        test_scores_list.sort()
        print(dataset, np.mean(test_scores_list[5:-5]), np.std(test_scores_list[5:-5]))
