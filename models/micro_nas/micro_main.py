"""Entry point."""

import argparse

import torch

import models.micro.trainer as trainer
import utils

from models.common.common_main  import register_default_args

def build_args():
    parser = argparse.ArgumentParser(description='GraphNAS')
    register_default_args(parser)
    parser.add_argument('--supervise', type=bool, default=False)
    parser.add_argument('--num_of_cell', type=int, default=4)  # dest="num_of_cell for micro space"
    args = parser.parse_args()

    return args
def main(args):  # pylint:disable=redefined-outer-name
    # args.dataset = "Cora"
    # args.supervise = True
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

