from models.macro_nas.pyg.pyg_gnn_model_manager import GeoCitationManager
from models.simple_nas.simple_gnn import SimpleGNN


class SimpleCitationManager(GeoCitationManager):

    def build_gnn(self, actions):
        model = SimpleGNN(actions, self.in_feats, self.n_classes, drop_out=self.args.in_drop, multi_label=False,
                          batch_normal=False, residual=False, state_num=2)
        return model
