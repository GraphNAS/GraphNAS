"""Entry point."""

import argparse

import torch

import graphnas_variants.simple_graphnas.simple_trainer as trainer
from graphnas import utils

from graphnas.main import register_default_args


def build_args():
    parser = argparse.ArgumentParser(description='GraphNAS')
    register_default_args(parser)
    args = parser.parse_args()

    return args


def main(args):  # pylint:disable=redefined-outer-name
    args.search_mode = "simple"
    args.format = "simple"
    # args.controller_max_step = 0
    args.submanager_log_file = "_simple_sub_manager_logger_file.txt"
    print(args)
    if args.cuda and not torch.cuda.is_available():  # cuda is not available
        args.cuda = False

    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    utils.makedirs(args.dataset)

    trnr = trainer.SimpleTrainer(args)

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
