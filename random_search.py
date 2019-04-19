import sys

sys.path.extend(['/home/gaoyang/PycharmProjects/GraphNAS-simple', '/home/gaoyang/PycharmProjects/GraphNAS-simple'])

from models.gnn_controller import state_space
import random
from main import build_args_for_ppi as build_ppi_args
from models.geo.geo_gnn_ppi_manager import GeoPPIManager


def generate_couple_structure(layer=2):
    action_list = []
    for _ in range(layer):
        for action in state_space:
            action_list.append(random.choice(state_space[action]))
    return action_list


def random_search_for_ppi():
    args = build_ppi_args()
    args.param_file = "random_ppi.pkl"
    args.epochs = 2
    controller = GeoPPIManager(args)
    best_structure = ""
    best_val_acc = 0
    for i in range(1000):
        print("structure %d" % i)
        actions = generate_couple_structure(2)
        format_ = "two"
        try:
            result = controller.train(actions, format_)
            if result:
                _, val_acc = result
            else:
                continue
        except RuntimeError as e:
            if "cuda" in str(e) or "CUDA" in str(e):
                print(e)
                val_acc = 0
            else:
                raise e
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_structure = actions
        if i % 1000 == 0:
            print(best_structure)


if __name__ == "__main__":
    random_search_for_ppi()
