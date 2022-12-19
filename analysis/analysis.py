import json
from copy import deepcopy

import pandas as pd

def read_file(source):

    results = []
    with open(source, "r") as f:
        lines = f.readlines()

    print(lines[1])
    print(lines[1])

    for line in lines:
        actions = line[:line.index(";")]

        actions_dataform = str(actions).strip("'<>() ").replace('\'', '\"')
        actions_struct = {}
        actions_struct = json.loads(actions_dataform)

        val_score = line.split(";")[-1]
        val_dataform = str(val_score).strip("'<>() ").replace('\'', '\"')
        val_struct = {}
        val_struct = json.loads(val_dataform)

        results.append((actions_struct, val_struct))

    return results

def return_df(nr_layers=2):
    f = read_file('/Users/teodorareu/PycharmProjects/GraphNAS_Project/FirstGeneratedCiteseer.txt')

    # dfl = []
    # for line in range(len(f)):
    #     l = f[line][0].append(f[line][1])
    #     dfl.append(f[line][0].append(f[line][1]))

    df = pd.DataFrame(f, columns=['actions', 'val'])
    df[['layer_1', 'agg_1', 'acc_1', 'att_1', 'hidd_dim_1', 'layer_2', 'agg_2', 'acc_2', 'att_2', 'hidd_dim_2']] = pd.DataFrame(df.actions.tolist(), index=df.index)
    return df

def analise_over_action(action_name, df):
    return df[[action_name,'val']].groupby(action_name).mean()

def analise_over_combinations_of_actions(actions, df):
    l = deepcopy(actions)
    actions.append('val')
    print(l)
    df = df[actions].groupby(l).mean()
    return df

def test():
    f = read_file('/Users/teodorareu/PycharmProjects/GraphNAS_Project/FirstGeneratedCiteseer.txt')
    l = return_df(2)

    A = analise_over_combinations_of_actions(['layer_1','layer_2'],l)
