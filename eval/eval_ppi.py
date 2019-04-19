import sys

from models.gnn_ppi_manager import build_args, PPIGCN

sys.path.extend(['../'])


if __name__ == '__main__':
    # graphnas
    actions = ['bilinear', 'max', 'tanh', 6, 64, 'gat_sym', 'mean', 'relu', 6, 128, 'gat', 'mlp',
               'linear', 16, 121]
    # random
    actions = ['linear', 'sum', 'relu', 16, 32,
               'linear', 'max', 'sigmoid', 6, 256,
               'gcn', 'max', 'linear', 1, 121]
    actions = ['bilinear', 'mlp', 'softplus', 16, 128,
               'gcn', 'mlp', 'relu', 8, 128,
               'linear', 'sum', 'linear', 1, 121]
    args = build_args()
    args.retrain_filename = args.param_file = "random_model_ppi.pk"
    args.optim_file = "opt_ppi.pk"
    args.retrain_epochs = 200
    controller = PPIGCN(args)
    controller.retrain(actions, with_param=False)
