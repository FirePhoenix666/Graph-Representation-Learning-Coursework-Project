import os
import csv
import numpy as np
import random
import argparse
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import remove_self_loops, add_self_loops
from MI.kde import mi_kde
from models.GCNII import GCNII


# Set parameters used in building model and group norm
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="Cora", help='{Cora, Citeseer}')
parser.add_argument('--layer', type=int, default=2, help='Number of layers.')
parser.add_argument('--type_norm', type=str, default="group", help='{None, batch, group, pair}')
parser.add_argument('--dropout', type=float, default=0.6)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--lamda', type=float, default=0.5)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--num_groups', type=int, default=10)
args = parser.parse_args()

if args.layer == 2:
    skip_weight = 0.005
else:
    skip_weight = 0.001

if args.dataset == 'Cora':
    num_features = 1433
    num_classes = 7
elif args.dataset == 'Citeseer':
    num_features = 3703
    num_classes = 6


# Calculate instance information gain
def compute_MI(model, data):
    model.eval()
    data_x = data.x.data.cpu().numpy()
    with torch.no_grad():
        layers_self = model()
    layer_self = layers_self.data.cpu().numpy()
    MI_XiX = mi_kde(layer_self, data_x, var=0.1)
    return MI_XiX


# Calculate group distance
def dis_cluster(model, data):
    model.eval()
    with torch.no_grad():
        X = model()
    X_labels = []
    for i in range(num_classes):
        X_label = X[data.y == i].data.cpu().numpy()
        h_norm = np.sum(np.square(X_label), axis=1, keepdims=True)
        h_norm[h_norm == 0.] = 1e-3
        X_label = X_label / np.sqrt(h_norm)
        X_labels.append(X_label)

    dis_intra = 0.
    for i in range(num_classes):
        x2 = np.sum(np.square(X_labels[i]), axis=1, keepdims=True)
        dists = x2 + x2.T - 2 * np.matmul(X_labels[i], X_labels[i].T)
        dis_intra += np.mean(dists)
    dis_intra /= num_classes

    dis_inter = 0.
    for i in range(num_classes - 1):
        for j in range(i + 1, num_classes):
            x2_i = np.sum(np.square(X_labels[i]), axis=1, keepdims=True)
            x2_j = np.sum(np.square(X_labels[j]), axis=1, keepdims=True)
            dists = x2_i + x2_j.T - 2 * np.matmul(X_labels[i], X_labels[j].T)
            dis_inter += np.mean(dists)
    num_inter = float(num_classes * (num_classes - 1) / 2)
    dis_inter /= num_inter

    print('dis_intra: ', dis_intra)
    print('dis_inter: ', dis_inter)
    return dis_intra, dis_inter


def load_data(dataset="Cora"):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', dataset)
    if dataset in ["Cora", "Citeseer"]:
        # Import data via Planetoid API
        data = Planetoid(path, dataset, split='random', transform=T.NormalizeFeatures())[0]
        # Preprocess edge indexes in case of wrong data format
        # Details: https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html
        num_nodes = data.x.size(0)
        edge_index, _ = remove_self_loops(data.edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=num_nodes)
        if isinstance(edge_index, tuple):
            data.edge_index = edge_index[0]
        else:
            data.edge_index = edge_index
        return data
    else:
        raise Exception(f'the dataset of {dataset} has not been implemented')


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# Define seeds used for generating random numbers
seeds = [100, 200, 300, 400, 500]

# Load dataset
data = load_data(args.dataset)

# Training hyperparameter
weight_decay1 = 0.01
weight_decay2 = 5e-4
lr = 0.01
patience = 100
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define empty lists to store results of each seed
acc_test_list = []
MI_XiX_seeds = []
dis_ratio_seeds = []


def train():
    # Train the model and return loss
    model.train()
    optimizer.zero_grad()
    loss_train = F.nll_loss(model()[data.train_mask], data.y[data.train_mask])
    loss_train.backward()
    optimizer.step()
    return loss_train.item()

def test():
    # Evaluate the model and return accuracies
    model.eval()
    with torch.no_grad():
        logits = model()
    loss_val = F.nll_loss(logits[data.val_mask], data.y[data.val_mask]).item()
    for _, mask in data('val_mask'):
        pred = logits[mask].max(1)[1]
        val_accs = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

    for _, mask in data('test_mask'):
        pred = logits[mask].max(1)[1]
        test_accs = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    return loss_val, val_accs, test_accs

for seed in seeds:
    set_seed(seed)
    # Load model and data
    model = GCNII(args, skip_weight, num_features, num_classes, data).to(device)
    data = data.to(device)
    # Set optimizer
    optimizer = torch.optim.Adam([
        dict(params=model.reg_params, weight_decay=weight_decay1),
        dict(params=model.non_reg_params, weight_decay=weight_decay2)], lr=lr)

    # Initialize best_val_loss with a large number 9999999
    best_val_loss = 9999999
    best_val_acc = 0.
    test_acc = 0
    bad_counter = 0
    best_epoch = 0
    epoch_data = []

    for epoch in range(1, 1001):
        loss_tra = train()
        loss_val, acc_val_tmp, acc_test_tmp = test()
        # Save the best model
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            test_acc = acc_test_tmp
            bad_counter = 0
            best_epoch = epoch

        log = 'Epoch: {:03d}, Train loss: {:.4f}, Val loss: {:.4f}, Test acc: {:.4f}'
        epoch_data.append([epoch, loss_tra, loss_val, test_acc])
        print(log.format(epoch, loss_tra, loss_val, test_acc))

    dis_intra, dis_inter = dis_cluster(model, data)
    dis_ratio = dis_inter / dis_intra
    # if both dis_inter and dis_intra are close to zero, the value of dis_ratio is nan
    # in this case, we assign the distance ratio to 1.
    dis_ratio = 1. if np.isnan(dis_ratio) else dis_ratio
    dis_ratio_seeds.append(dis_ratio)
    acc_test_list.append(test_acc)
    MI_XiX = compute_MI(model, data)
    MI_XiX_seeds.append(MI_XiX)
    avg_MI_XiX = np.mean(MI_XiX_seeds)
    avg_dis_ratio = np.mean(dis_ratio_seeds)
    # Save the result under current seed
    acc_test_list.append(test_acc)
    print('best Epoch: {:03d}, Val loss: {:.4f}, Test acc: {:.4f}'.format(best_epoch, best_val_loss, test_acc))
    filename = f'GCNII_{args.dataset}_{args.type_norm}_L{args.layer}_S{seed}'
    with open(filename + ".csv", 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Epoch', 'Loss', 'Validation Loss', 'Test Accuracy'])

        for row in epoch_data:
            csv_writer.writerow(row)


print('test acc of 5 seeds: ', acc_test_list)
print('avg test acc and std: ', np.mean(acc_test_list), np.std(acc_test_list))
print(f'experiment results of {args.type_norm} applied in GCNII on dataset {args.dataset}')
print('instance information gain: ', avg_MI_XiX)
print('group distance ratio: ', avg_dis_ratio)
