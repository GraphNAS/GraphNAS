from graphnas_variants.micro_graphnas.micro_search_space import gnn_list


class SimpleSearchSpace(object):
    def __init__(self, search_space=None):
        if search_space:
            self.search_space = search_space
        else:
            self.search_space = {
                "conv_type": gnn_list,
                "out_dim": [8, 16, 32, 64, 128, 256]
            }

    def get_search_space(self):
        return self.search_space

    def generate_action_list(self, num_of_layers=2):
        action_names = list(self.search_space.keys())
        action_list = action_names * num_of_layers
        return action_list
