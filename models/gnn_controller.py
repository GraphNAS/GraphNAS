"""A module with NAS controller-related code."""

import torch
import torch.nn.functional as F

import utils

state_space = {
    # "bn": ["yes", "no"],
    "attention_type": ["gat", "gcn", "cos", "const", "gat_sym", 'linear', 'generalized_linear'],
    'aggregator_type': ["sum", "mean", "max", "mlp", ],  # remove lstm
    'activate_function': ["sigmoid", "tanh", "relu", "linear",
                          "softplus", "leaky_relu", "relu6", "elu"],
    'number_of_heads': [1, 2, 4, 6, 8, 16],
    'hidden_units': [4, 8, 16, 32, 64, 128, 256],
    # 'jump_knowledge_type': ['none'],
}


def _construct_action(actions, state_space, skip_conn=False):
    state_length = len(state_space)
    layers = []
    for action in actions:
        predicted_actions = []
        keys = list(state_space.keys())
        for index, each in enumerate(action):
            state_index = index % state_length
            if skip_conn:
                if state_index == 0:
                    predicted_actions.append(each)  # skip_conn
                else:
                    state_index -= 1
                    predicted_actions.append(state_space[keys[state_index]][each])
            else:
                predicted_actions.append(state_space[keys[state_index]][each])
        layers.append(predicted_actions)
    return layers


class GNNNASController(torch.nn.Module):
    """Based on
    https://github.com/pytorch/examples/blob/master/word_language_model/model.py

    TODO(brendan): RL controllers do not necessarily have much to do with
    language models.

    Base the controller RNN on the GRU from:
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
    """

    def __init__(self, args, num_layers=3, skip_conn=False, controller_hid=100, cuda=True, mode="train",
                 softmax_temperature=5.0, tanh_c=2.5):
        torch.nn.Module.__init__(self)
        self.mode = mode
        self.num_layers = num_layers
        self.skip_conn = skip_conn
        self.controller_hid = controller_hid
        self.is_cuda = cuda

        if args and args.softmax_temperature:
            self.softmax_temperature = args.softmax_temperature
        else:
            self.softmax_temperature = softmax_temperature
        if args and args.tanh_c:
            self.tanh_c = args.tanh_c
        else:
            self.tanh_c = tanh_c

        self.num_tokens = []
        state_space_length = []

        if not skip_conn:
            keys = state_space.keys()
            for key in keys:
                state_space_length.append(len(state_space[key]))
            for _ in range(self.num_layers):
                self.num_tokens += state_space_length
        else:
            keys = state_space.keys()
            for idx in range(1, self.num_layers + 1):
                self.num_tokens += [idx]
                for key in keys:
                    self.num_tokens += len(state_space[key])

        num_total_tokens = sum(self.num_tokens)

        self.encoder = torch.nn.Embedding(num_total_tokens,
                                          controller_hid)
        self.lstm = torch.nn.LSTMCell(controller_hid, controller_hid)

        self.decoders = []
        if not skip_conn:
            # share the same decoder
            for idx, size in enumerate(state_space_length):
                decoder = torch.nn.Linear(controller_hid, size)
                self.decoders.append(decoder)
        else:
            # TODO share decoder for same actions
            # TODO Test1

            state_decoder = []  # shared decoder
            for idx, size in enumerate(state_space_length):
                decoder = torch.nn.Linear(controller_hid, size)
                state_decoder.append(decoder)

            for idx in range(1, self.num_layers + 1):
                # skip_connection
                decoder = torch.nn.Linear(controller_hid, idx)
                self.decoders.append(decoder)
                # common action
                for decoder in state_decoder:
                    self.decoders.append(decoder)
            # old version
            # for idx, size in enumerate(self.num_tokens):
            #     decoder = torch.nn.Linear(controller_hid, size)
            #     self.decoders.append(decoder)

        self._decoders = torch.nn.ModuleList(self.decoders)

        self.reset_parameters()
        self.static_init_hidden = utils.keydefaultdict(self.init_hidden)

        def _get_default_hidden(key):
            return utils.get_variable(
                torch.zeros(key, controller_hid),
                cuda,
                requires_grad=False)

        self.static_inputs = utils.keydefaultdict(_get_default_hidden)

    def reset_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        for decoder in self.decoders:
            decoder.bias.data.fill_(0)

    def forward(self,  # pylitorch.nn.Embedding(num_total_tokens,nt:disable=arguments-differ
                inputs,
                hidden,
                block_idx,
                is_embed):
        if not is_embed:
            embed = self.encoder(inputs)
        else:
            embed = inputs

        hx, cx = self.lstm(embed, hidden)
        logits = self.decoders[block_idx](hx)

        logits /= self.softmax_temperature

        # exploration
        if self.mode == 'train':
            logits = (self.tanh_c * F.tanh(logits))

        return logits, (hx, cx)

    def sample(self, batch_size=1, with_details=False, save_dir=None):
        """Samples a set of `args.num_blocks` many computational nodes from the
        controller, where each node is made up of an activation function, and
        each node except the last also includes a previous node.
        """
        if batch_size < 1:
            raise Exception(f'Wrong batch_size: {batch_size} < 1')

        # [B, L, H] zeros
        inputs = self.static_inputs[batch_size]
        hidden = self.static_init_hidden[batch_size]

        entropies = []
        log_probs = []
        actions = []
        # NOTE(brendan): The RNN controller alternately outputs an activation,
        # followed by a previous node, for each block except the last one,
        # which only gets an activation function. The last node is the output
        # node, and its previous node is the average of all leaf nodes.
        for block_idx in range(len(self.num_tokens)):
            if not self.skip_conn:
                decoder_index = block_idx % len(state_space)
            else:
                decoder_index = block_idx
            logits, hidden = self.forward(inputs,
                                          hidden,
                                          decoder_index,
                                          is_embed=(block_idx == 0))

            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            # TODO(brendan): .mean() for entropy?
            entropy = -(log_prob * probs).sum(1, keepdim=False)

            action = probs.multinomial(num_samples=1).data
            selected_log_prob = log_prob.gather(
                1, utils.get_variable(action, requires_grad=False))

            # TODO(brendan): why the [:, 0] here? Should it be .squeeze(), or
            # .view()? Same below with `action`.
            entropies.append(entropy)
            log_probs.append(selected_log_prob[:, 0])

            inputs = utils.get_variable(
                action[:, 0] + sum(self.num_tokens[:block_idx]),
                self.is_cuda,
                requires_grad=False)

            actions.append(action[:, 0])

        actions = torch.stack(actions).transpose(0, 1)
        dags = _construct_action(actions, state_space)

        if with_details:
            return dags, torch.cat(log_probs), torch.cat(entropies)

        return dags

    def init_hidden(self, batch_size):
        zeros = torch.zeros(batch_size, self.controller_hid)
        return (utils.get_variable(zeros, self.is_cuda, requires_grad=False),
                utils.get_variable(zeros.clone(), self.is_cuda, requires_grad=False))


if __name__ == "__main__":
    cntr = GNNNASController(None, cuda=False)
    print(cntr.sample(200, with_details=True))
