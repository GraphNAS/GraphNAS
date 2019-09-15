import torch


def build_adj(edge_index, num_nodes):
    """
    for undirected graph
    :param edge_index:
    :param num_nodes:
    :return:
    """
    if num_nodes is None:
        num_nodes = max(edge_index[0]) + 1
    edge_attr = torch.ones(edge_index.size(1), dtype=torch.float)
    size = torch.Size([num_nodes, num_nodes])
    adj = torch.sparse_coo_tensor(edge_index, edge_attr, size)
    eye = torch.arange(start=0, end=num_nodes)
    eye = torch.stack([eye, eye])
    eye = torch.sparse_coo_tensor(eye, torch.ones([num_nodes]), size)
    adj = adj.t() + adj + eye  # greater than 1 when edge_index is already symmetrical

    adj = adj.to_dense().gt(0).to_sparse().type(torch.float)

    return adj


def get_degree(edge_index, num_nodes):
    adj = build_adj(edge_index, num_nodes)
    degree = adj.to_dense().sum(1)
    return degree


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.uint8, device=index.device)
    mask[index] = 1
    return mask


def random_labels_splits(data, num_classes, required_labels=20):
    # Set new random planetoid splits:
    # * required_labels * num_classes labels for training
    # * 500 labels for validation
    # * 1000 labels for testing

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:required_labels] for i in indices], dim=0)

    rest_index = torch.cat([i[required_labels:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:500], size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index[500:1500], size=data.num_nodes)

    return data


def nas_labels_splits(data, num_classes, required_labels=20):
    # Set new random planetoid splits:
    # * 20 * num_classes labels for training
    # * 500 labels for validation
    # * 1000 labels for testing

    indices = []
    visible_data_length = len(data.y) - 1000

    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
    data.test_mask[data.num_nodes - 1000:] = 1

    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[index.lt(visible_data_length)]
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:required_labels] for i in indices], dim=0)

    rest_index = torch.cat([i[required_labels:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:500], size=data.num_nodes)

    return data


def nas_labels_by_degree(data, num_classes, required_labels=20, descending=False):
    if hasattr(data, "degree"):
        degree = data.degree
    else:
        data.degree = get_degree(data.edge_index, None)
        degree = data.degree
    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index_degree = degree[index]
        index = index[index_degree.sort(dim=0, descending=descending).indices]  # sort by degree
        indices.append(index)

    train_index = torch.cat([i[:required_labels] for i in indices], dim=0)

    rest_index = torch.cat([i[required_labels:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:500], size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index[500:1500], size=data.num_nodes)

    return data


def fix_size_split(data, train_num=1000, val_num=500, test_num=1000):
    assert train_num + val_num + test_num <= data.x.size(0), "not enough data"
    indices = torch.randperm(data.x.size(0))

    data.train_mask = index_to_mask(indices[:train_num], size=data.num_nodes)
    data.val_mask = index_to_mask(indices[train_num:train_num + val_num], size=data.num_nodes)
    data.test_mask = index_to_mask(indices[-test_num:], size=data.num_nodes)

    return data


def fix_size_split_by_degree(data, train_num=1000, val_num=500, test_num=1000, descending=False):
    assert train_num + val_num + test_num <= data.x.size(0), "not enough data"

    if hasattr(data, "degree"):
        degree = data.degree
    else:
        data.degree = get_degree(data.edge_index, None)
        degree = data.degree
    indices = degree.sort(dim=0, descending=descending).indices
    train_index = indices[:train_num]

    rest_index = indices[train_num:]
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[train_num:train_num + val_num], size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index[-test_num:], size=data.num_nodes)

    return data


def shuffle_test_data(data):
    assert hasattr(data, "train_mask") and hasattr(data, "val_mask"), "require train_mask and val_mask"

    used_data = data.train_mask + data.val_mask
    # rest_index = used_data.nonzero().view(-1)
    rest_index = used_data.lt(1).nonzero().view(-1)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.test_mask = index_to_mask(rest_index[:1000], size=data.num_nodes)

    return data


def val_labels_by_degree(data, num_classes, required_train_labels=20, required_val_labels=20, descending=False):
    assert hasattr(data, "train_mask"), "require train_mask"
    if hasattr(data, "degree"):
        degree = data.degree
    else:
        data.degree = get_degree(data.edge_index, None)
        degree = data.degree
    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index_degree = degree[index]
        index = index[index_degree.sort(dim=0, descending=descending).indices]  # sort by degree
        indices.append(index)

    train_index = torch.cat([i[:required_train_labels] for i in indices], dim=0)
    val_index = torch.cat([i[required_train_labels: required_train_labels + required_val_labels] for i in indices],
                          dim=0)
    rest_index = torch.cat([i[required_train_labels + required_val_labels:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index[:1000], size=data.num_nodes)

    return data
