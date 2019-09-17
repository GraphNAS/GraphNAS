import numpy as np
import torch

# sys.path.extend(['/GraphNAS'])
from graphnas_variants.macro_graphnas.pyg.pyg_gnn_model_manager import GeoCitationManager
from eval_scripts.semi.eval_designed_gnn import build_args
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)


if __name__ == "__main__":
    args = build_args()

    gnn_list = [
        ['generalized_linear', 'sum', 'leaky_relu', 6, 16, 'gat', 'sum', 'leaky_relu', 1, 7],
        ['gat_sym', 'sum', 'linear', 4, 256, 'cos', 'sum', 'softplus', 2, 6],
        ['const', 'sum', 'relu', 4, 32, 'generalized_linear', 'sum', 'leaky_relu', 8, 3]
    ]
    dataset_list =["Cora", "Citeseer", "Pubmed"]
    for dataset, actions in zip(dataset_list, gnn_list):
        args.dataset = dataset
        manager = GeoCitationManager(args)
        test_scores_list = []
        for i in range(100):
            val_acc, test_acc = manager.evaluate(actions)
            test_scores_list.append(test_acc)
        print("_" * 80)
        test_scores_list.sort()
        print(dataset, np.mean(test_scores_list[5:-5]), np.std(test_scores_list[5:-5]))
