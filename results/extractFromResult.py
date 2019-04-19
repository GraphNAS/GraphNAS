import heapq

import numpy as np


def extract_from_citation():
    filename = "citetion_result.txt"
    with open(filename, 'r') as f:
        lines = f.readlines()

    best_acc = 0
    msg = []
    for line in lines:
        score = line.split(",")[-1]
        score = float(score)
        if score >= best_acc:
            if score > best_acc:
                msg.clear()
            best_acc = score
            msg.append(line)
    sturctures = set()
    for each in msg:
        right = "]"
        index = each.index(right)
        sturctures.add(each[:index + 1])
    print(best_acc)
    for each in sturctures:
        print(each)


def extract_from_cora():
    filename = "nohup.txt"
    with open(filename) as f:
        lines = f.readlines()
    count = 0
    for i, line in enumerate(lines):
        if "training controller over" in line:
            count += 1
            if count == 10:
                break
    lines = lines[:i]
    results = []
    actions = ""
    val_scores = ""
    test_scores = ""
    for line in lines:
        if "train action:" in line:
            actions = line.split(":")[-1]
            val_scores = 0
        if "val_acc" in line:
            line = line.split()
            # print(line)
            tmp_val_score = float(line[-4])
            tmp_test_scores = float(line[-1])
            if tmp_val_score > val_scores:
                val_scores = tmp_val_score
                test_scores = tmp_test_scores
        if "Top 10 average:" in line:
            results.append([actions, val_scores, test_scores])
    results.sort(key=lambda x: x[1], reverse=True)


def extract_from_ppi(filename="test_with_new_dgl.txt"):
    with open(filename) as f:
        lines = f.readlines()

    results = []
    actions = ""
    val_scores = ""
    test_scores = ""
    for line in lines:
        if "train action:" in line:
            actions = line.split(":")[-1]
            val_scores = 0
        if "val_acc" in line:
            line = line.split(":")
            # print(line)
            tmp_val_score = float(line[3].split(",")[0])
            tmp_test_scores = float(line[4][:6])
            if tmp_val_score > val_scores:
                val_scores = tmp_val_score
                test_scores = tmp_test_scores
        if "Top 10 average:" in line:
            results.append([actions, val_scores, test_scores])
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def extract_from_eval(filename="test_with_new_dgl.txt"):
    with open(filename) as file:
        lines = file.readlines()

    results = []
    actions = ""
    val_score = ""
    all = []
    for line in lines:
        if "train action:" in line:
            actions = line.split(":")[-1]
            if results:
                all.append(results)
                results = []
        if "Graph 11" in line:
            index = line.rfind(":")
            test_score = line[index + 1:]
            test_score = float(test_score)
            if actions and test_score:
                results.append([actions, test_score])
            else:
                pass
                # print(val_score)
    # results.sort(key=lambda x:x[1],reverse=True)
    all.append(results)
    return all


def extract_from_gat(filename="gat.txt"):
    # filename = "val_sturcture.txt"
    with open(filename) as f:
        lines = f.readlines()

    results = []
    epoch = 0
    val_score = 0
    for line in lines:
        if "Graph: 0011" in line:
            val_score = float(line.split()[-2][:-1])
            results.append(val_score)
    return results


def top(results, k=1):
    scores = [float(each[1][:-1]) for each in results]
    top_list = []
    for i in range(1, len(scores) + 1):
        max_scores = heapq.nlargest(k, scores[:i])
        top_list.append(np.mean(max_scores))
    return top_list


def average_score(results, k=1):
    scores = [float(each[1][:-1]) for each in results]
    average_list = []
    for i in range(k, len(scores) + 1):
        average_list.append(np.mean(scores[i - k:i]))
    return average_list


def max_score(results, k=1):
    scores = [float(each[1][:-1]) for each in results]
    average_list = []
    for i in range(k, len(scores) + 1):
        average_list.append(np.max(scores[i - k:i]))
    return average_list
