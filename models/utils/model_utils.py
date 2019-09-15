import numpy as np


class FixedList(list):
    def __init__(self, size=10):
        super(FixedList, self).__init__()
        self.size = size

    def append(self, obj):
        if len(self) >= self.size:
            self.pop(0)
        super().append(obj)


class TopAverage(object):
    def __init__(self, top_k=10):
        self.scores = []
        self.top_k = top_k

    def get_top_average(self):
        if len(self.scores) > 0:
            return np.mean(self.scores)
        else:
            return 0

    def get_average(self, score):
        if len(self.scores) > 0:
            avg = np.mean(self.scores)
        else:
            avg = 0
        # print("Top %d average: %f" % (self.top_k, avg))
        self.scores.append(score)
        self.scores.sort(reverse=True)
        self.scores = self.scores[:self.top_k]
        return avg

    def get_reward(self, score):
        reward = score - self.get_average(score)
        return np.clip(reward, -0.5, 0.5)


class EarlyStop(object):
    def __init__(self, size=10):
        self.size = size
        self.train_loss_list = FixedList(size)
        self.train_score_list = FixedList(size)
        self.val_loss_list = FixedList(size)
        self.val_score_list = FixedList(size)

    def should_stop(self, train_loss, train_score, val_loss, val_score):
        flag = False
        if len(self.train_loss_list) < self.size:
            pass
        else:
            if val_loss > 0:
                if val_loss >= np.mean(self.val_loss_list):  # and val_score <= np.mean(self.val_score_list)
                    flag = True
            elif train_loss > np.mean(self.train_loss_list):
                flag = True
        self.train_loss_list.append(train_loss)
        self.train_score_list.append(train_score)
        self.val_loss_list.append(val_loss)
        self.val_score_list.append(val_score)

        return flag

    def should_save(self, train_loss, train_score, val_loss, val_score):
        if len(self.val_loss_list) < 1:
            return False
        if train_loss < min(self.train_loss) and val_score > max(self.val_score_list):
            # if val_loss < min(self.val_loss_list) and val_score > max(self.val_score_list):
            return True
        else:
            return False


def process_action(actions, type, args):

    if type == 'two':
        actual_action = actions
        # actual_action[-2] = args.num_class
        actual_action[-1] = args.num_class
        # actual_action.append( args.num_class)

        return actual_action

    elif type == "simple":
        actual_action = actions
        index = len(actual_action) - 1
        actual_action[index]["out_dim"] = args.num_class

        return actual_action

    elif type == "dict":
        return actions

    elif type == "micro":
        return actions


def calc_f1(output, labels, sigmoid=True):
    y_true = labels.cpu().data.numpy()
    y_pred = output.cpu().data.numpy()
    from sklearn import metrics
    if not sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")
