class GNNManager(object):
    # build model and set hyperparameters
    def __init__(self, args):
        self.args = args

    def load_param(self):
        raise NotImplementedError

    def save_param(self):
        raise NotImplementedError

    # exploration, or train shared parameters
    def train(self, actions):
        raise NotImplementedError

    # evaluate model from scratch
    def retrain(self, actions):
        raise NotImplementedError

    # evaluate model with a few training or no training
    def test_with_param(self, actions):
        raise NotImplementedError
