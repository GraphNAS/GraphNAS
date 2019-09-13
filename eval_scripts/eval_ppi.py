import sys

from models.gnn_ppi_manager import build_args, PPIGNN

sys.path.extend(['../'])


if __name__ == '__main__':

    actions = ['linear', 'sum', 'relu', 16, 32,
               'linear', 'max', 'sigmoid', 6, 256,
               'gcn', 'max', 'linear', 1, 121]

    args = build_args()
    args.retrain_filename = args.param_file = "ppi.pk"
    args.retrain_epochs = 200
    controller = PPIGNN(args)
    controller.retrain(actions, with_param=False)
