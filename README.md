# Graph Representation Learning Group Project 

## Introduction

The report is attached in file 54751_48372_56111_48119_project_report.pdf  

We apply three types of normalization methods and DropEdge technique on GCN, GAT and GCNII models, in order to mitigate the over-smoothing and over-fitting problems. 

This project is based on torch and torch-geometric.

## Requirements

```
cudnn==7.6.5
networkx==2.8.4
nltk==3.8.1
numpy==1.19.2
tensorflow-gpu==2.4.1
torch==1.12.0+cu113
torch-cluster==1.6.0+pt112cu113
torch-geometric==2.3.0
torch-scatter==2.1.0+pt112cu113
torch-sparse==0.6.14
torch-spline-conv==1.2.1+pt112cu113
```

## Normalization Block

Three types of normalization methods are sealed in `models\norm_block.py`.

`models\GCN.py`, `models\GAT.py` and `models\GCNII.py` all import the normalization block, and build norm layers within the model architectures.

## DropEdge Block

DropEdge technique is sealed in `models/dropedge_block.py`.

DropEdge is essectially a data augementation technique, randomly dropping edge and preserving a certain proportion of edges. 

This technique is imported by `trainer.py` and `trainer_GCNII.py`, which both load the dataset.

## Train 

### Train GCN and GAT

To train GCN and GAT models, measure the group distance ratio as well as instance information gain, run:

```
python main.py --type_model='GCN' --type_norm='batch' --dataset='Citeseer' --dropedge_percent=0.05
```

Hyperparameter explanations:

--`type_norm`: the type of normalization layer. We include ['None', 'batch', 'pair', 'group'] for none normalization,
batch normalization, pair normalization and differentiable group normalization, respectively.

--`type_model`: the type of GNN model. We include ['GCN', 'GAT']

--`dataset`: we include ['Cora', 'Citeseer']

--`dropedge_percent`: the perserving rate of edges, continuously ranging from 0. to 1.. The value of 1. corresponds to the original dataset. The value of 0.05 means randomly perserving 5% of original edges in the training set.

### Train GCNII

To train GCNII model, run:
```
python trainer_GCNII.py --type_norm='batch' --dataset='Cora' --layer=15
```
where `type_norm` choices are ['None', 'batch', 'pair', 'group'], `dataset` choices are ['Cora', 'Citeseer'], and `layer` choices are [2, 15, 30].

## Result

Our results is saved in `result\` folder. Please find the training logs in `result\logs\` and the numerial results in `result\Cora_csv\` and `result\Citeseer_csv\`.

We plot the losses using `plot.ipynb` and save the images in `result\plot_Cora\` folder.
