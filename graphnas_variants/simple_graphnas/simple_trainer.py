from graphnas.trainer import Trainer
from graphnas_variants.simple_graphnas.simple_model_manager import SimpleCitationManager


class SimpleTrainer(Trainer):

    def build_model(self):

        if self.args.search_mode == "simple":
            self.submodel_manager = SimpleCitationManager(self.args)

            from graphnas_variants.simple_graphnas.simple_search_space import SimpleSearchSpace
            search_space_cls = SimpleSearchSpace()
            self.search_space = search_space_cls.get_search_space()
            self.action_list = search_space_cls.generate_action_list(self.args.layers_of_child_model)
            # build RNN controller
            from graphnas.graphnas_controller import SimpleNASController
            self.controller = SimpleNASController(self.args, action_list=self.action_list,
                                                  search_space=self.search_space,
                                                  cuda=self.args.cuda)
            # self.controller = NASController(self.args, cuda=self.args.cuda,
            #                                       num_layers=self.args.layers_of_child_model)
        if self.cuda:
            self.controller.cuda()

    def form_gnn_info(self, gnn):
        gnn_list = [gnn]
        state_length = len(self.search_space)
        result_gnn = []
        for gnn_info in gnn_list:
            predicted_gnn = {}
            gnn_layer_info = {}
            for index, each in enumerate(gnn_info):
                if index % state_length == 0:  # current layer information is over
                    if gnn_layer_info:
                        predicted_gnn[index // state_length - 1] = gnn_layer_info
                        gnn_layer_info = {}
                gnn_layer_info[self.action_list[index]] = gnn_info[index]
            predicted_gnn[index // state_length] = gnn_layer_info  # add the last layer info
            result_gnn.append(predicted_gnn)
        return result_gnn[0]

    @property
    def model_info_filename(self):
        return f"{self.args.dataset}_{self.args.search_mode}_{self.args.format}_results.txt"
