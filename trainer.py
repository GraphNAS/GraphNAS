import glob
import os

import numpy as np
import scipy.signal
import torch.nn.parallel
from torch import nn

import models
import utils
from models.gnn_citation_manager import CitationGNN
from models.geo.geo_gnn_citation_manager import GeoCitationManager
from models.geo.geo_gnn_ppi_manager import GeoPPIManager
logger = utils.get_logger()


def discount(x, amount):
    return scipy.signal.lfilter([1], [1, -amount], x[::-1], axis=0)[::-1]


def _get_optimizer(name):
    if name.lower() == 'sgd':
        optim = torch.optim.SGD
    elif name.lower() == 'adam':
        optim = torch.optim.Adam

    return optim


class Trainer(object):
    """Manage the training process"""

    def __init__(self, args):
        """
        Constructor for training algorithm.
        Build sub-model(shared) and controller.
        Build optimizer and cross entropy loss for controller.
        Optimizer and loss function managed by sub-model.

        Args:
            args: From command line, picked up by `argparse`.
        """
        self.args = args
        self.controller_step = 0  # counter for controller
        self.cuda = args.cuda
        self.epoch = 0
        self.shared_step = 0  # counter for sub_model
        self.start_epoch = 0

        # remove regulation

        self.max_length = self.args.shared_rnn_max_length

        self.with_retrain = True
        self.shared = None
        self.controller = None
        self.build_model()  # build controller and sub-model

        controller_optimizer = _get_optimizer(self.args.controller_optim)
        self.controller_optim = controller_optimizer(self.controller.parameters(), lr=self.args.controller_lr)

        if self.args.load_path:
            self.load_model()

        self.ce = nn.CrossEntropyLoss()

    def build_model(self):

        if self.args.search_mode == "nas":
            self.args.share_param = False
            self.with_retrain = True
            self.args.shared_initial_step = 0
            logger.info("NAS-like mode: retrain without share param")
            pass

        if self.args.dataset in ["cora", "citeseer", "pubmed"]:
            self.shared = CitationGNN(self.args)
            self.controller = models.GNNNASController(self.args, cuda=self.args.cuda, num_layers=2)

        if self.args.dataset in ["Cora", "Citeseer", "Pubmed"]:
            self.shared = GeoCitationManager(self.args)
            self.controller = models.GNNNASController(self.args, cuda=self.args.cuda, num_layers=2)

        if self.args.dataset == "PPI":
            self.shared = GeoPPIManager(self.args)
            self.controller = models.GNNNASController(self.args, cuda=self.args.cuda, num_layers=3)

        if self.cuda:
            self.controller.cuda()

    def train(self):
        """
        Each epoch consists of two phase:
        - In the first phase, shared parameters are trained to exploration.
        - In the second phase, the controller's parameters are trained.
        """

        for self.epoch in range(self.start_epoch, self.args.max_epoch):
            # 1. Training the shared parameters of the child models
            self.train_shared(max_step=self.args.shared_initial_step)
            # 2. Training the controller parameters theta
            self.train_controller()

            if self.epoch % self.args.save_epoch == 0:
                self.save_model()

        best_actions = self.derive()
        print("best structure:" + str(best_actions))
        print(self.evaluate(best_actions))
        self.save_model()

    def train_shared(self, max_step=50, action=None):
        """
        Args:
            max_step: Used to run extra training steps as a warm-up.
            action: If not None, is used instead of calling sample().

        """
        if max_step == 0:  # no train shared
            return
        print("*" * 35, "training model", "*" * 35)
        structure_list = action if action else self.controller.sample(
            max_step)

        for action in structure_list:
            try:
                _, val_score = self.shared.train(action)
                logger.info(f"{action}, val_score:{val_score}")
            except RuntimeError as e:
                if 'CUDA' in str(e):  # usually CUDA Out of Memory
                    print(e)
                else:
                    raise e
            self.shared_step += 1

        print("*" * 35, "training over", "*" * 35)

    def get_reward(self, structure_list, entropies, hidden):
        """
        Computes the reward of a single sampled model on validation data.
        """
        if not isinstance(entropies, np.ndarray):
            entropies = entropies.data.cpu().numpy()
        if not isinstance(structure_list[0], list):
            structure_list = [structure_list]
        reward_list = []  # reward list
        for actions in structure_list:

            reward = self.shared.test_with_param(actions, with_retrain=self.with_retrain)
            if reward is None:  # cuda error hanppened
                reward = 0
            else:
                reward = reward[1]

            reward_list.append(reward)

        if self.args.entropy_mode == 'reward':
            rewards = reward_list + self.args.entropy_coeff * entropies
        elif self.args.entropy_mode == 'regularizer':
            rewards = reward_list * np.ones_like(entropies)
        else:
            raise NotImplementedError(f'Unkown entropy mode: {self.args.entropy_mode}')

        return rewards, hidden

    def train_controller(self):
        """
            Train controller to find better structure.
        """
        print("*" * 35, "training controller", "*" * 35)
        model = self.controller
        model.train()

        avg_reward_base = None
        baseline = None
        adv_history = []
        entropy_history = []
        reward_history = []

        hidden = self.controller.init_hidden(self.args.batch_size)
        total_loss = 0
        for step in range(self.args.controller_max_step):
            # sample models
            structure_list, log_probs, entropies = self.controller.sample(
                with_details=True)

            # calculate reward
            np_entropies = entropies.data.cpu().numpy()
            results = self.get_reward(structure_list, np_entropies, hidden)
            torch.cuda.empty_cache()

            if results:  # has reward
                rewards, hidden = results
            else:
                continue  # CUDA Error happens, drop structure and step into next iteration

            # discount
            if 1 > self.args.discount > 0:
                rewards = discount(rewards, self.args.discount)

            reward_history.extend(rewards)
            entropy_history.extend(np_entropies)

            # moving average baseline
            if baseline is None:
                baseline = rewards
            else:
                decay = self.args.ema_baseline_decay
                baseline = decay * baseline + (1 - decay) * rewards

            adv = rewards - baseline
            adv_history.extend(adv)

            adv = utils.get_variable(adv, self.cuda, requires_grad=False)
            # policy loss
            loss = -log_probs * adv
            if self.args.entropy_mode == 'regularizer':
                loss -= self.args.entropy_coeff * entropies

            loss = loss.sum()  # or loss.mean()

            # update
            self.controller_optim.zero_grad()
            loss.backward()

            if self.args.controller_grad_clip > 0:
                torch.nn.utils.clip_grad_norm(model.parameters(),
                                              self.args.controller_grad_clip)
            self.controller_optim.step()

            total_loss += utils.to_item(loss.data)

            self.controller_step += 1
            torch.cuda.empty_cache()

        print("*" * 35, "training controller over", "*" * 35)

    def evaluate(self, action):
        """
        Evaluate a structure on the validation set.
        """
        self.controller.eval()

        results = self.shared.retrain(action)
        if results:
            reward, scores = results
        else:
            return

        logger.info(f'eval | {action} | loss: {reward:8.2f} | ppl: {scores:8.2f}')

    def derive(self, sample_num=None):
        """
        sample a serial of structures, and return the best structure.
        """

        if sample_num is None:
            sample_num = self.args.derive_num_sample

        structure_list, _, entropies = self.controller.sample(sample_num, with_details=True)

        max_R = 0
        best_actions = None
        filename = f"{self.args.dataset}_{self.args.search_mode}_results.txt"
        for actions in structure_list:
            results = self.get_reward(actions, entropies, None)
            if results:
                R, _ = results
            else:
                continue
            if R.max() > max_R:
                max_R = R.max()
                best_actions = actions
            with open(filename, "a") as f:
                msg = f"actions:{actions},reward:{R}\n"
                f.write(msg)

        logger.info(f'derive | max_R: {max_R:8.6f}')

        return best_actions

    @property
    def controller_path(self):
        return f'{self.args.dataset}/controller_epoch{self.epoch}_step{self.controller_step}.pth'

    @property
    def controller_optimizer_path(self):
        return f'{self.args.dataset}/controller_epoch{self.epoch}_step{self.controller_step}_optimizer.pth'

    def get_saved_models_info(self):
        paths = glob.glob(os.path.join(self.args.dataset, '*.pth'))
        paths.sort()

        def get_numbers(items, delimiter, idx, replace_word, must_contain=''):
            return list(set([int(
                name.split(delimiter)[idx].replace(replace_word, ''))
                for name in basenames if must_contain in name]))

        basenames = [os.path.basename(path.rsplit('.', 1)[0]) for path in paths]
        epochs = get_numbers(basenames, '_', 1, 'epoch')
        shared_steps = get_numbers(basenames, '_', 2, 'step', 'shared')
        controller_steps = get_numbers(basenames, '_', 2, 'step', 'controller')

        epochs.sort()
        shared_steps.sort()
        controller_steps.sort()

        return epochs, shared_steps, controller_steps

    def save_model(self):

        torch.save(self.controller.state_dict(), self.controller_path)
        torch.save(self.controller_optim.state_dict(), self.controller_optimizer_path)

        logger.info(f'[*] SAVED: {self.controller_path}')

        epochs, shared_steps, controller_steps = self.get_saved_models_info()

        for epoch in epochs[:-self.args.max_save_num]:
            paths = glob.glob(
                os.path.join(self.args.dataset, f'*_epoch{epoch}_*.pth'))

            for path in paths:
                utils.remove_file(path)

    def load_model(self):
        epochs, shared_steps, controller_steps = self.get_saved_models_info()

        if len(epochs) == 0:
            logger.info(f'[!] No checkpoint found in {self.args.dataset}...')
            return

        self.epoch = self.start_epoch = max(epochs)
        self.shared_step = max(shared_steps)
        self.controller_step = max(controller_steps)

        if self.args.num_gpu == 0:
            map_location = lambda storage, loc: storage
        else:
            map_location = None

        self.controller.load_state_dict(
            torch.load(self.controller_path, map_location=map_location))
        self.controller_optim.load_state_dict(
            torch.load(self.controller_optimizer_path, map_location=map_location))
        logger.info(f'[*] LOADED: {self.controller_path}')


