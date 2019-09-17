from graphnas_variants.macro_graphnas.pyg import GeoCitationManager
from graphnas_variants.simple_graphnas.simple_gnn import SimpleGNN


class SimpleCitationManager(GeoCitationManager):

    def build_gnn(self, actions):
        model = SimpleGNN(actions, self.in_feats, self.n_classes, drop_out=self.args.in_drop, multi_label=False,
                          batch_normal=False, residual=False, state_num=2)
        return model
