import numpy as np
import torch
from eval_scripts.sup.eval_designed_gnn import build_args
# sys.path.extend(['/GraphNAS'])
from graphnas_variants.micro_graphnas.micro_model_manager import MicroCitationManager
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)


if __name__ == "__main__":
    args = build_args()
    args.cuda = True
    gnn_list = [
        # Cora:
        {'action': [1, 'sage', 1, 'gat_6', 2, 'sage', 0, 'gat_8', 'elu', 'add'], 'hyper_param': [0.0005, 0.7, 5e-05, 512]},
        # Citeseer:
        {'action': [0, 'cheb', 0, 'zero', 'relu', 'add'], 'hyper_param': [0.0005, 0.8, 1e-05, 128]},
        # Pubmed:
        {'action': [0, 'linear', 1, 'gat_6', 'elu', 'product'], 'hyper_param': [0.01, 0.2, 0, 128]},
    ]
    dataset_list = [ "Cora", "Citeseer", "Pubmed" ] #
    for shuffle in [True]: # False
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
            print(dataset, np.mean(test_scores_list[5:-5]), np.std(test_scores_list[5:-5]))
