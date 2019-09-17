"""Entry point."""

import argparse

import torch

import graphnas_variants.micro_graphnas.micro_trainer as trainer
from graphnas import utils
from graphnas.main import register_default_args


def build_args():
    parser = argparse.ArgumentParser(description='GraphNAS')
    register_default_args(parser)
    parser.add_argument('--predict_hyper', type=bool, default=True)
    parser.add_argument('--num_of_cell', type=int, default=2)  # dest="num_of_cell for micro space"
    args = parser.parse_args()

    return args


def main(args):  # pylint:disable=redefined-outer-name
    print(args)
    if args.cuda and not torch.cuda.is_available():  # cuda is not available
        args.cuda = False

    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    utils.makedirs(args.dataset)

    trnr = trainer.HyperTrainer(args)

    if args.mode == 'train':
        print(args)
        trnr.train()
    elif args.mode == 'derive':
        trnr.derive()
    else:
        raise Exception(f"[!] Mode not found: {args.mode}")


if __name__ == "__main__":
    args = build_args()
    main(args)
