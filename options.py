import argparse

def reset_weight(args):
    if args.dataset == 'Citeseer':
        args.skip_weight = 0.001 if args.num_layers < 6 else 0.005
    elif args.dataset == 'Cora':
        if args.type_model in ['GCN']:
            args.skip_weight = 0.001 if args.num_layers < 6 else 0.03
        elif args.type_model in ['GAT']:
            args.skip_weight = 0.001 if args.num_layers < 6 else 0.01
    return args

class BaseOptions():

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""

    def initialize(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--random_seed', type=int, default=123)
        parser.add_argument("--cuda", type=bool, default=True, required=False, help="run in cuda mode")
        parser.add_argument('--cuda_num', type=int, default=0, help="GPU number")
        parser.add_argument("--dataset", type=str, default="Cora", required=False, help="The input dataset.")

        # Build up the network hyperparameter
        parser.add_argument('--type_model', type=str, default="GCN")
        parser.add_argument('--layers', type=int, default=3)
        parser.add_argument("--epochs", type=int, default=1000, help="number of training the one shot model")
        parser.add_argument("--dropout", type=float, default=0.6, help="input feature dropout")
        parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
        parser.add_argument('--weight_decay', type=float, default=5e-4)
        parser.add_argument('--grad_clip', type=float, default=0.0)
        parser.add_argument('--dim_hidden', type=int, default=16)

        # Settings for normalization and droping edge
        parser.add_argument('--type_norm', type=str, default="None", help='{None, batch, group, pair}')
        parser.add_argument('--dropedge_percent', type=float, default=1.0, help='sampling percent in DropEdge')
        parser.add_argument('--num_groups', type=int, default=10)
        parser.add_argument('--skip_weight', type=float, default=0.005)
        parser.add_argument('--loss_weight', type=float, default=0.)

        args = parser.parse_args()
        args = self.reset_model_parameter(args)
        return args

    def reset_model_parameter(self, args):
        if args.dataset == 'Cora':
            args.num_feats = 1433
            args.num_classes = 7
            args.dropout = 0.6
            args.lr = 0.005
        elif args.dataset == 'Citeseer':
            args.num_feats = 3703
            args.num_classes = 6
            args.dropout = 0.6
            args.weight_decay = 5e-5
            args.lr = 0.005
        else:
            raise Exception(f'Please include the num_feats, num_classes of {args.dataset} first')
        return args