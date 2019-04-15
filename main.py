"""Entry point."""

import argparse

import torch

import trainer
import utils


def build_args():
    parser = argparse.ArgumentParser(description='GraphNAS')
    # register_data_args(parser)
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'derive'],
                        help='train: Training ENAS, derive: Deriving Architectures')
    parser.add_argument('--random_seed', type=int, default=123)
    parser.add_argument("--cuda", type=bool, default=True, required=False,
                        help="run in cuda mode")
    parser.add_argument('--save_epoch', type=int, default=5)
    parser.add_argument('--max_save_num', type=int, default=5)
    # controller
    parser.add_argument('--layers_of_child_model', type=int, default=3)
    parser.add_argument('--shared_initial_step', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--entropy_mode', type=str, default='reward', choices=['reward', 'regularizer'])
    parser.add_argument('--entropy_coeff', type=float, default=1e-4)
    parser.add_argument('--shared_rnn_max_length', type=int, default=35)
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--search_mode', type=str, choices=['enas', 'nas', 'graphnas'],
                        default='nas')
    parser.add_argument('--max_epoch', type=int, default=1000)
    # NOTE(brendan): irrelevant for actor critic.
    parser.add_argument('--ema_baseline_decay', type=float, default=0.95)  # TODO: very important
    parser.add_argument('--discount', type=float, default=1.0)  # TODO
    parser.add_argument('--controller_max_step', type=int, default=5,  # TODO 50
                        help='step for controller parameters')
    parser.add_argument('--controller_optim', type=str, default='adam')
    parser.add_argument('--controller_lr', type=float, default=3.5e-4,
                        help="will be ignored if --controller_lr_cosine=True")
    parser.add_argument('--controller_grad_clip', type=float, default=0)
    parser.add_argument('--tanh_c', type=float, default=2.5)
    parser.add_argument('--softmax_temperature', type=float, default=5.0)
    parser.add_argument('--derive_num_sample', type=int, default=100)

    # child model
    parser.add_argument("--dataset", type=str, default="Cora", required=False,
                        help="The input dataset.")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--retrain_epochs", type=int, default=200,
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


def build_args_for_ppi():
    args = build_args()
    args.dataset = "PPI"
    if args.layers_of_child_model < 3:
        args.layers_of_child_model = 3
    args.in_feats = 50
    args.num_class = 121
    args.in_drop = 0
    args.weight_decay = 0
    args.epochs = 50
    return args


def main(args):  # pylint:disable=redefined-outer-name

    if args.cuda and not torch.cuda.is_available():  # cuda is not available
        args.cuda = False

    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    utils.makedirs(args.dataset)

    trnr = trainer.Trainer(args)

    if args.mode == 'train':
        print(args)
        trnr.train()
    elif args.mode == 'derive':
        trnr.derive()
    else:
        raise Exception(f"[!] Mode not found: {args.mode}")


if __name__ == "__main__":
    args = build_args()
    if args.dataset == "PPI":
        args = build_args_for_ppi()
    main(args)
