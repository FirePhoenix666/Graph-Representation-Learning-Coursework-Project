
python main.py --type_model='GCN' --type_norm='None' --dataset='Cora' --dropedge_percent=1.
python main.py --type_model='GCN' --type_norm='None' --dataset='Citeseer' --dropedge_percent=1.
python main.py --type_model='GAT' --type_norm='None' --dataset='Cora' --dropedge_percent=1.
python main.py --type_model='GAT' --type_norm='None' --dataset='Citeseer' --dropedge_percent=1.

python main.py --type_model='GCN' --type_norm='None' --dataset='Cora' --dropedge_percent=0.05
python main.py --type_model='GCN' --type_norm='None' --dataset='Citeseer' --dropedge_percent=0.05
python main.py --type_model='GAT' --type_norm='None' --dataset='Cora' --dropedge_percent=0.05
python main.py --type_model='GAT' --type_norm='None' --dataset='Citeseer' --dropedge_percent=0.05

python main.py --type_model='GCN' --type_norm='batch' --dataset='Cora' --dropedge_percent=1.
python main.py --type_model='GCN' --type_norm='batch' --dataset='Citeseer' --dropedge_percent=1.
python main.py --type_model='GAT' --type_norm='batch' --dataset='Cora' --dropedge_percent=1.
python main.py --type_model='GAT' --type_norm='batch' --dataset='Citeseer' --dropedge_percent=1.

python main.py --type_model='GCN' --type_norm='batch' --dataset='Cora' --dropedge_percent=0.05
python main.py --type_model='GCN' --type_norm='batch' --dataset='Citeseer' --dropedge_percent=0.05
python main.py --type_model='GAT' --type_norm='batch' --dataset='Cora' --dropedge_percent=0.05
python main.py --type_model='GAT' --type_norm='batch' --dataset='Citeseer' --dropedge_percent=0.05

python main.py --type_model='GCN' --type_norm='pair' --dataset='Cora' --dropedge_percent=1.
python main.py --type_model='GCN' --type_norm='pair' --dataset='Citeseer' --dropedge_percent=1.
python main.py --type_model='GAT' --type_norm='pair' --dataset='Cora' --dropedge_percent=1.
python main.py --type_model='GAT' --type_norm='pair' --dataset='Citeseer' --dropedge_percent=1.

python main.py --type_model='GCN' --type_norm='pair' --dataset='Cora' --dropedge_percent=0.05
python main.py --type_model='GCN' --type_norm='pair' --dataset='Citeseer' --dropedge_percent=0.05
python main.py --type_model='GAT' --type_norm='pair' --dataset='Cora' --dropedge_percent=0.05
python main.py --type_model='GAT' --type_norm='pair' --dataset='Citeseer' --dropedge_percent=0.05

python main.py --type_model='GCN' --type_norm='group' --dataset='Cora' --dropedge_percent=1.
python main.py --type_model='GCN' --type_norm='group' --dataset='Citeseer' --dropedge_percent=1.
python main.py --type_model='GAT' --type_norm='group' --dataset='Cora' --dropedge_percent=1.
python main.py --type_model='GAT' --type_norm='group' --dataset='Citeseer' --dropedge_percent=1.

python main.py --type_model='GCN' --type_norm='group' --dataset='Cora' --dropedge_percent=0.05
python main.py --type_model='GCN' --type_norm='group' --dataset='Citeseer' --dropedge_percent=0.05
python main.py --type_model='GAT' --type_norm='group' --dataset='Cora' --dropedge_percent=0.05
python main.py --type_model='GAT' --type_norm='group' --dataset='Citeseer' --dropedge_percent=0.05

python trainer_GCNII.py --type_norm='None' --dataset='Cora' --layer=2
python trainer_GCNII.py --type_norm='None' --dataset='Cora' --layer=15
python trainer_GCNII.py --type_norm='None' --dataset='Cora' --layer=30
python trainer_GCNII.py --type_norm='None' --dataset='Citeseer' --layer=2
python trainer_GCNII.py --type_norm='None' --dataset='Citeseer' --layer=15
python trainer_GCNII.py --type_norm='None' --dataset='Citeseer' --layer=30

python trainer_GCNII.py --type_norm='batch' --dataset='Cora' --layer=2
python trainer_GCNII.py --type_norm='batch' --dataset='Cora' --layer=15
python trainer_GCNII.py --type_norm='batch' --dataset='Cora' --layer=30
python trainer_GCNII.py --type_norm='batch' --dataset='Citeseer' --layer=2
python trainer_GCNII.py --type_norm='batch' --dataset='Citeseer' --layer=15
python trainer_GCNII.py --type_norm='batch' --dataset='Citeseer' --layer=30

python trainer_GCNII.py --type_norm='pair' --dataset='Cora' --layer=2
python trainer_GCNII.py --type_norm='pair' --dataset='Cora' --layer=15
python trainer_GCNII.py --type_norm='pair' --dataset='Cora' --layer=30
python trainer_GCNII.py --type_norm='pair' --dataset='Citeseer' --layer=2
python trainer_GCNII.py --type_norm='pair' --dataset='Citeseer' --layer=15
python trainer_GCNII.py --type_norm='pair' --dataset='Citeseer' --layer=30

python trainer_GCNII.py --type_norm='group' --dataset='Cora' --layer=2
python trainer_GCNII.py --type_norm='group' --dataset='Cora' --layer=15
python trainer_GCNII.py --type_norm='group' --dataset='Cora' --layer=30
python trainer_GCNII.py --type_norm='group' --dataset='Citeseer' --layer=2
python trainer_GCNII.py --type_norm='group' --dataset='Citeseer' --layer=15
python trainer_GCNII.py --type_norm='group' --dataset='Citeseer' --layer=30