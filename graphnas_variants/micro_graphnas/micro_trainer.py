from graphnas.trainer import Trainer, logger
from graphnas_variants.micro_graphnas.micro_search_space import IncrementSearchSpace


class HyperTrainer(Trainer):

    def build_model(self):
        self.args.format = "micro"
        if self.args.search_mode == "nas":
            self.args.share_param = False
            self.with_retrain = True
            self.args.shared_initial_step = 0
            logger.info("NAS-like mode: retrain without share param")
            pass
        if not hasattr(self.args, "num_of_cell"):
            self.args.num_of_cell = 2
        search_space_cls = IncrementSearchSpace()
        search_space = search_space_cls.get_search_space()
        from graphnas.graphnas_controller import SimpleNASController
        from graphnas_variants.micro_graphnas.micro_model_manager import MicroCitationManager
        self.submodel_manager = MicroCitationManager(self.args)
        self.search_space = search_space
        action_list = search_space_cls.generate_action_list(cell=self.args.num_of_cell)
        if hasattr(self.args, "predict_hyper") and self.args.predict_hyper:
            self.action_list = action_list + ["learning_rate", "dropout", "weight_decay", "hidden_unit"]
        else:
            self.action_list = action_list
        self.controller = SimpleNASController(self.args, action_list=self.action_list, search_space=self.search_space,
                                              cuda=self.args.cuda)
        if self.cuda:
            self.controller.cuda()

    def form_gnn_info(self, action):
        actual_action = {}
        if self.args.predict_hyper:
            actual_action["action"] = action[:-4]
            actual_action["hyper_param"] = action[-4:]
        else:
            actual_action["action"] = action
            actual_action["hyper_param"] = [0.005, 0.8, 5e-5, 128]
        return actual_action
