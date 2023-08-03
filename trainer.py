
import os
import csv
import torch
import numpy as np
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import remove_self_loops, add_self_loops

from MI.kde import mi_kde
from models.GCN import GCN
from models.GAT import GAT
from models.dropedge_block import Sampler



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


def evaluate(output, labels, mask):
    _, indices = torch.max(output, dim=1)
    correct = torch.sum(indices[mask] == labels[mask])
    return correct.item() * 1.0 / mask.sum().item()


class trainer(object):
    def __init__(self, args):
        # Load dataset
        self.dataset = args.dataset
        self.data = load_data(self.dataset)

        # Apply the negative log likelihood loss
        # Details: https://pytorch.org/docs/stable/generated/torch.nn.functional.nll_loss.html
        self.loss_fn = torch.nn.functional.nll_loss

        # Set the model type
        self.type_model = args.type_model
        if self.type_model == 'GCN':
            self.model = GCN(args)
        elif self.type_model == 'GAT':
            self.model = GAT(args)
        else:
            raise Exception(f'the model of {self.type_model} has not been implemented')
        
        # Hyperparameters for training models
        self.epochs = args.epochs
        self.grad_clip = args.grad_clip
        self.weight_decay = args.weight_decay
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.seed = args.random_seed
        self.type_norm = args.type_norm
        self.skip_weight = args.skip_weight

        # Choose the device to place the model and data
        self.device = torch.device(f'cuda:{args.cuda_num}' if args.cuda else 'cpu')
        self.data.to(self.device)
        self.model.to(self.device)

        # Set the sampler for DropEdge operation
        self.dropedge_percent = args.dropedge_percent
        self.sampler = Sampler(self.data, device = self.device)

    def train_compute_MI(self):
        best_acc = 0
        epoch_data = []
        # For each epoch
        for epoch in range(self.epochs):
            # Excute DropEdge first, jumpting to models/dropedge_blocks.py
            (data_x, data_edge_index) = self.sampler.randomedge_sampler(percent=self.dropedge_percent)
            # Train model
            loss_train, acc_train, acc_valid, acc_test = self.train_net(data_x, data_edge_index)
            print('Epoch: {:02d}, Loss: {:.4f}, val_acc: {:.4f}, test_acc:{:.4f}'.format(epoch, loss_train, acc_valid, acc_test))

            epoch_data.append([epoch, loss_train, acc_valid, acc_test])
            # Save the best model
            if best_acc < acc_valid:
                best_acc = acc_valid
                self.model.cpu()
                self.save_model(self.type_model, self.dataset)
                self.model.to(self.device)

            # Save data to a CSV file
            filename = self.filename(filetype='result_csv', type_model=self.type_model, dataset=self.dataset)
            with open(filename + ".csv", 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['Epoch', 'Loss', 'Validation Accuracy', 'Test Accuracy'])

                for row in epoch_data:
                    csv_writer.writerow(row)

        # Reload the best model
        state_dict = self.load_model(self.type_model, self.dataset)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

        # Evaluate the saved model
        acc_train, acc_valid, acc_test = self.run_testSet()
        print('val_acc: {:.4f}, test_acc:{:.4f}'.format(acc_valid, acc_test))

        # Compute the instance information gain
        MI_XiX = self.compute_MI()

        # Compute the intra-group and inter-group distances first
        dis_intra, dis_inter = self.dis_cluster()
        # Obtain the group distance ratio
        # Note: If both dis_inter and dis_intra are close to zero, the value of dis_ratio is nan
        #       in this case, we assign the distance ratio to 1.
        dis_ratio = dis_inter / dis_intra
        dis_ratio = 1. if np.isnan(dis_ratio) else dis_ratio

        return acc_test, MI_XiX, dis_ratio
    
    def train_net(self, data_x, data_edge_index):
        # Training phase
        loss_train = self.run_trainSet(data_x, data_edge_index)
        # Evaluation phase
        acc_train, acc_valid, acc_test = self.run_testSet()
        return loss_train, acc_train, acc_valid, acc_test
    
    def run_trainSet(self, data_x, data_edge_index):
        # Train the model and return loss
        self.model.train()
        logits = self.model(data_x, data_edge_index)
        logits = F.log_softmax(logits[self.data.train_mask], 1)
        loss = self.loss_fn(logits, self.data.y[self.data.train_mask])
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        return loss.item()

    def run_testSet(self):
        # Evaluate the model and return accuracies
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.data.x, self.data.edge_index)
        logits = F.log_softmax(logits, 1)
        acc_train = evaluate(logits, self.data.y, self.data.train_mask)
        acc_valid = evaluate(logits, self.data.y, self.data.val_mask)
        acc_test = evaluate(logits, self.data.y, self.data.test_mask)
        return acc_train, acc_valid, acc_test

    def compute_MI(self):
        # Compute the instance information gain
        self.model.eval()
        data_x = self.data.x.data.cpu().numpy()
        with torch.no_grad():
            layers_self = self.model(self.data.x, self.data.edge_index)
        layer_self = layers_self.data.cpu().numpy()
        MI_XiX = mi_kde(layer_self, data_x, var=0.1)
        return MI_XiX
    
    def dis_cluster(self):
        # Compute the intra-group and inter-group distances
        self.model.eval()
        with torch.no_grad():
            X = self.model(self.data.x, self.data.edge_index)
        X_labels = []
        for i in range(self.model.num_classes):
            X_label = X[self.data.y == i].data.cpu().numpy()
            h_norm = np.sum(np.square(X_label), axis=1, keepdims=True)
            h_norm[h_norm == 0.] = 1e-3
            X_label = X_label / np.sqrt(h_norm)
            X_labels.append(X_label)

        dis_intra = 0.
        for i in range(self.model.num_classes):
            x2 = np.sum(np.square(X_labels[i]), axis=1, keepdims=True)
            dists = x2 + x2.T - 2 * np.matmul(X_labels[i], X_labels[i].T)
            dis_intra += np.mean(dists)
        dis_intra /= self.model.num_classes

        dis_inter = 0.
        for i in range(self.model.num_classes-1):
            for j in range(i+1, self.model.num_classes):
                x2_i = np.sum(np.square(X_labels[i]), axis=1, keepdims=True)
                x2_j = np.sum(np.square(X_labels[j]), axis=1, keepdims=True)
                dists = x2_i + x2_j.T - 2 * np.matmul(X_labels[i], X_labels[j].T)
                dis_inter += np.mean(dists)
        num_inter = float(self.model.num_classes * (self.model.num_classes-1) / 2)
        dis_inter /= num_inter

        print('dis_intra: ', dis_intra)
        print('dis_inter: ', dis_inter)
        return dis_intra, dis_inter
    
    def load_model(self, type_model='GCN', dataset='Cora'):
        filename = self.filename(filetype='params', type_model=type_model, dataset=dataset)
        if os.path.exists(filename+".pth.tar"):
            print('load model: ', type_model, filename+".pth.tar")
            return torch.load(filename+".pth.tar")
        else:
            return None

    def save_model(self, type_model='GCN', dataset='Cora'):
        filename = self.filename(filetype='params', type_model=type_model, dataset=dataset)
        state = self.model.state_dict()
        torch.save(state, filename+".pth.tar")
        print('save model to', filename+".pth.tar")

    def filename(self, filetype='logs', type_model='GCN', dataset='PPI'):
        filedir = f'./{filetype}/{dataset}'
        if not os.path.exists(filedir):
            os.makedirs(filedir)

        num_layers = int(self.model.num_layers)
        type_norm = self.type_norm
        dropedge_percent = int(self.dropedge_percent*100)
        seed = int(self.seed)

        if type_norm == 'group':
            group = self.model.num_groups
            skip_weight = int(self.model.skip_weight * 1e3)

            filename = f'{filetype}_{type_model}_{type_norm}' \
                f'L{num_layers}D{dropedge_percent}S{seed}G{group}S{skip_weight}'
        else:

            filename = f'{filetype}_{type_model}_{type_norm}' \
                       f'L{num_layers}D{dropedge_percent}S{seed}'

        filename = os.path.join(filedir, filename)
        return filename