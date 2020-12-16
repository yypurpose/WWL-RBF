# -----------------------------------------------------------------------------
# This file contains several utility functions for reproducing results 
# of the WWL paper
#
# October 2019, M. Togninalli
# -----------------------------------------------------------------------------
import numpy as np
import os
import igraph as ig
from scipy.sparse import csr_matrix

import torch
# import dgl
# from dgl.data import TUDataset
# from torch_geometric.datasets import TUDataset
from myTUDataset import TUDataset
from torch_geometric.transforms import ToDense, ToSparseTensor

from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.model_selection._validation import _fit_and_score
from sklearn.base import clone
from sklearn.metrics import make_scorer, accuracy_score

#################
# File loaders 
#################

def load_continuous_graphs(dataset_name):
    data = TUDataset(dataset_name)
    graphs, labels = zip(*[data[i] for i in range(len(data))])
    labels = torch.tensor(labels).numpy()

    # initialize
    node_labels = [] # discrete label for each node in each graph
    node_features = [] # continuous attributes for each node in each graph
    adj_mat = [] # adjency matrix of each graph
    n_nodes = [] # node number in each graph
    edge_features = [] # edge weights in each graph

    # Iterate across graphs and load initial node features
    for graph in graphs:
        # Load features
        node_labels.append(graph.ndata['node_labels'].numpy().astype(float))
        if graph.ndata.get('node_attr') != None:
            node_features.append(graph.ndata['node_attr'].numpy().astype(float))
        else:
            node_features.append(graph.ndata['node_labels'].numpy().astype(float))
        adj_mat.append(graph.adj().to_dense().numpy())
        n_nodes.append(graph.num_nodes())
        if graph.edata.get('node_labels') != None:
            # Edge features
            edges_s = graph.edges(form='all')[0].numpy()
            edges_e = graph.edges(form='all')[1].numpy()
            edges_weights = graph.edata['node_labels'].numpy().reshape(-1)
            weight_cur = csr_matrix((edges_weights, (edges_s,edges_e))).toarray()
            edge_features.append(weight_cur)

    n_nodes = np.asarray(n_nodes)
    node_labels = np.asarray(node_labels)
    node_features = np.asarray(node_features)
    edge_features = np.asarray(edge_features)

    return node_labels, node_features, adj_mat, n_nodes, edge_features, labels

def load_matrices(directory):
    '''
    Loads all the wasserstein matrices in the directory.
    '''
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory,f))]
    wass_matrices = []
    hs = []
    for f in sorted(files):
        hs.append(int(f.split('.npy')[0].split('it')[-1])) # Hoping not to have h > 9 !
        wass_matrices.append(np.load(os.path.join(directory,f)))
    return wass_matrices, hs

##################
# Graph processing
##################

# def create_adj_avg(adj_cur):
# 	'''
# 	create adjacency
# 	'''
# 	deg = np.sum(adj_cur, axis = 1)
# 	deg = np.asarray(deg).reshape(-1)

# 	deg[deg!=1] -= 1

# 	deg = 1/deg
# 	deg_mat = np.diag(deg)
# 	adj_cur = adj_cur.dot(deg_mat.T).T
	
# 	return adj_cur

def create_weight_avg(adj_cur, weight_cur=None):
    deg = np.sum(adj_cur, axis = 1)
    deg = np.asarray(deg).reshape(-1)

    deg[deg!=1] -= 1

    deg = 1/deg
    deg_mat = np.diag(deg)
    if weight_cur is None:
        adj_cur = adj_cur.dot(deg_mat.T).T
        return adj_cur
    else:
        weight_cur = weight_cur.dot(deg_mat.T).T 
        return weight_cur

def create_labels_seq_cont(node_features, adj_mat, h, edge_features=None):
    '''
    create label sequence for continuously attributed graphs
    '''
    n_graphs = len(node_features)
    labels_sequence = []
    for i in range(n_graphs):
        graph_feat = []

        for it in range(h+1):
            if it == 0:
                graph_feat.append(node_features[i])
            else:
                adj_cur = adj_mat[i]+np.identity(adj_mat[i].shape[0])
                # adj_cur = create_adj_avg(adj_cur)
                if edge_features is None:
                    weight_cur = create_weight_avg(adj_cur)
                else:
                    weight_cur = create_weight_avg(adj_cur, edge_features[i])

                np.fill_diagonal(weight_cur, 0)
                graph_feat_cur = 0.5*(np.dot(weight_cur, graph_feat[it-1]) + graph_feat[it-1])
                graph_feat.append(graph_feat_cur)

        labels_sequence.append(np.concatenate(graph_feat, axis = 1))
        if i % 100 == 0:
            print(f'Processed {i} graphs out of {n_graphs}')
	
    return labels_sequence


#######################
# Hyperparameter search
#######################

def custom_grid_search_cv(model, param_grid, precomputed_kernels, y, cv=5):
    '''
    Custom grid search based on the sklearn grid search for an array of precomputed kernels
    '''
    # 1. Stratified K-fold
    cv = StratifiedKFold(n_splits=cv, shuffle=False)
    results = []
    for train_index, test_index in cv.split(precomputed_kernels[0], y):
        split_results = []
        params = [] # list of dict, its the same for every split
        # run over the kernels first
        for K_idx, K in enumerate(precomputed_kernels):
            # Run over parameters
            for p in list(ParameterGrid(param_grid)):
                sc = _fit_and_score(clone(model), K, y, scorer=make_scorer(accuracy_score), 
                        train=train_index, test=test_index, verbose=0, parameters=p, fit_params=None)
                split_results.append(sc)
                params.append({'K_idx': K_idx, 'params': p})
        results.append(split_results)
    # Collect results and average
    results = np.array(results)
    fin_results = results.mean(axis=0)
    # select the best results
    best_idx = np.argmax(fin_results)
    # Return the fitted model and the best_parameters
    ret_model = clone(model).set_params(**params[best_idx]['params'])
    return ret_model.fit(precomputed_kernels[params[best_idx]['K_idx']], y), params[best_idx]
