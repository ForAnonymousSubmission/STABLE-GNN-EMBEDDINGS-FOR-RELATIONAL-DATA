import os
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp
import torch
from torch import Tensor
import torch_geometric
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import Planetoid
import networkx as nx
from networkx.algorithms import community
import torch.nn as nn 
import pandas as pd
from torch_geometric.utils import add_self_loops, negative_sampling
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import GATConv, GINConv
import random
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import  degree
import random
from tqdm import tqdm
import copy

class GAT_GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, embedding_dim):
        super(GAT_GNN, self).__init__()
        # GCN layers to compute the embeddings
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, embedding_dim)
    
    def forward(self, x, edge_index):
        # First GCN layer with ReLU activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # Second GCN layer (this gives the embeddings)
        x = self.conv2(x, edge_index)
        return x

class CONV_GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, embedding_dim):
        super(CONV_GNN, self).__init__()
        # GCN layers to compute the embeddings
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, embedding_dim)
    
    def forward(self, x, edge_index):
        # First GCN layer with ReLU activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # Second GCN layer (this gives the embeddings)
        x = self.conv2(x, edge_index)
        return x

class MLP(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, out_channels):
        super(MLP, self).__init__()
        # Define the MLP layers
        self.fc1 = torch.nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, out_channels)
    
    def forward(self, x):
        # MLP forward pass
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)  # No softmax here as it's included in CrossEntropyLoss
        return x
    



class Order_one_GNN(MessagePassing):
    def __init__(self, in_dim, out_dim, nbr_filter_taps=2, *args, **kwargs) -> None:
        super().__init__()
        self.model_parameters = nn.ParameterList()
        for _ in range(nbr_filter_taps):
            H = nn.Parameter(torch.rand(in_dim, out_dim), requires_grad=True)
            self.model_parameters.append(H)

    def forward(self, X, edge_index):
        X1 = torch.matmul(X, self.model_parameters[0])
        X2 = torch.matmul(X, self.model_parameters[1])
        row, col = edge_index
        deg = degree(col, X.size(0), dtype=X.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        X2 = self.propagate(edge_index, x=X2, norm=norm)
        X_out = X1 + X2
        return X_out
    
    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]
        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j



class Node_GNN(nn.Module):
    def __init__(self, d_F, nbr_layers, *args, **kwargs) -> None: # , is_normalized
        super().__init__()
        assert(len(d_F)!=0 and len(d_F) +1 == nbr_layers)
        #self.is_normalized = is_normalized

        dim_parameters = [(1, d_F[0])]
        for i in range(1, len(d_F)):
            dim_parameters.append((d_F[i-1], d_F[i]))
        dim_parameters.append((d_F[-1], 1))

        self.convs = torch.nn.ModuleList()
        for dims in dim_parameters:
            in_dim, out_dim = dims
            self.convs.append(Order_one_GNN(in_dim, out_dim))

    def forward(self, X, edge_index):
        X_out = torch.zeros_like(X)  # Create a new tensor for X_out
        for i in range(X.shape[1]):
            X_in = X[:, i].unsqueeze(1)  # Add an extra dimension to match the expected input
            for conv in self.convs[:-1]:
                X_in = conv(X_in,  edge_index)
                X_in = F.elu(X_in)
            X_in = self.convs[-1](X_in, edge_index)
            X_out[:, i] = X_in.squeeze(1)  # Remove the extra dimension
        return X_out
    



def train_gnn_mlp(graph, train_mask, test_mask, gnn_model, mlp_model, optimizer, epochs=50):
    """
    Train the GNN + MLP for node classification using cross-entropy loss.

    Args:
    - graph (Data): PyTorch Geometric Data object representing the graph with node features and labels.
    - train_mask (torch.Tensor): A boolean mask indicating which nodes are in the training set.
    - test_mask (torch.Tensor): A boolean mask indicating which nodes are in the test set.
    - gnn_model (torch.nn.Module): The GNN model that computes embeddings.
    - mlp_model (torch.nn.Module): The MLP model that performs the classification.
    - optimizer (torch.optim.Optimizer): The optimizer to use.
    - epochs (int): The number of training epochs.

    Returns:
    - gnn_model (torch.nn.Module): The trained GNN model.
    - mlp_model (torch.nn.Module): The trained MLP model.
    """
    criterion = torch.nn.CrossEntropyLoss()


    save_dict = {}
    for epoch in tqdm(range(epochs)):
        gnn_model.train()
        mlp_model.train()
        optimizer.zero_grad()

        # Step 1: Use the GNN to compute node embeddings
        embeddings = gnn_model(graph.x, graph.edge_index)
        
        # Step 2: Use the MLP to classify nodes based on the embeddings
        out = mlp_model(embeddings)
        
        # Step 3: Compute the loss only for the training nodes
        loss = criterion(out[train_mask], graph.y[train_mask])
        loss.backward()
        optimizer.step()

        # Evaluation on test set
        gnn_model.eval()
        mlp_model.eval()


    return gnn_model, mlp_model, save_dict

def return_acc(graph, gnn_model, mlp_model, test_mask):
    with torch.no_grad():
        embeddings = gnn_model(graph.x, graph.edge_index).detach()
        out = mlp_model(embeddings)
        _, pred = out.max(dim=1)
        correct = pred[test_mask].eq(graph.y[test_mask]).sum().item()
        acc = correct / test_mask.sum().item()
    return acc

def split_train_test(graph, train_ratio: float = 0.8):
    """
    Split the nodes of the graph into a training and test set.
    
    Args:
    - graph (Data): A PyTorch Geometric Data object representing the graph with node features and labels.
    - train_ratio (float): Ratio of nodes to be used in the training set.

    Returns:
    - train_mask (torch.Tensor): A boolean mask for the training set nodes.
    - test_mask (torch.Tensor): A boolean mask for the test set nodes.
    """
    num_nodes = graph.num_nodes
    indices = list(range(num_nodes))
    
    # Shuffle the indices
    random.shuffle(indices)
    
    # Compute the size of the train set
    train_size = int(train_ratio * num_nodes)
    
    # Split indices into training and testing
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # Create boolean masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_indices] = True
    test_mask[test_indices] = True
    
    return train_mask, test_mask


def dict_to_dataframe(dict_of_arrays, data):
    """
    Convert a dictionary of arrays into a concatenated DataFrame with an additional column that contains the key
    corresponding to each array.
    
    Args:
    - dict_of_arrays (dict): A dictionary where the keys are strings and the values are numpy arrays or lists.

    Returns:
    - pd.DataFrame: The concatenated DataFrame with an additional 'key' column.
    """
    # List to store individual DataFrames
    dataframes = []
    
    # Iterate over the dictionary
    for key, array in dict_of_arrays.items():
        # Convert the array to a DataFrame
        df = pd.DataFrame(array)
        
        # Add a column to store the key
        df['key'] = key
        df['label'] = data.y.numpy()
        
        # Append the DataFrame to the list
        dataframes.append(df)
    
    # Concatenate all DataFrames
    concatenated_df = pd.concat(dataframes, ignore_index=True)
    concatenated_df.columns = ['id1', 'id2', 'id3', 'epoch', 'label']
    return concatenated_df

def new_edge_index(data, index, num_node_initial):
    index_bar = [i for i in range(num_node_initial) if i not in index]
    mask = []
    for i in index_bar:
        to_suppr += torch.where(data.index == i).tolist()
    to_suppr = list(np.array(to_suppr)[:,1].reshape(-1))
    to_keep = [i for i in range(data.edge_index.shape[1]) if i not in to_suppr]
    edge_index = data.edge_index[:,to_keep].to(data.edge_index.dtype)
    return edge_index

def new_edge_index_without_mapping(data, to_suppr, num_node_initial):
    
    mask_nodes = torch.ones(num_node_initial, dtype=torch.bool)
    mask_nodes[to_suppr] = False 

    mask_edges = mask_nodes[data.edge_index[0]] & mask_nodes[data.edge_index[1]]

    edge_index = data.edge_index[:, mask_edges]
    
    return edge_index



def new_edge_index(data, to_keep, to_suppr, num_node_initial):
    mapping = {node: idx for idx, node in enumerate(to_keep)}

    mask_nodes = torch.ones(num_node_initial, dtype=torch.bool)
    mask_nodes[to_suppr] = False 
    mask_edges = mask_nodes[data.edge_index[0]] & mask_nodes[data.edge_index[1]]
    edge_index = data.edge_index[:, mask_edges].T  

    for i, edge in enumerate(edge_index):
        node1, node2 = edge
        edge_index[i, 0], edge_index[i, 1] = mapping[int(node1)], mapping[int(node2)]

    return edge_index.T  



def make_expe(name, miss_rate = [0.0,0.1,0.3,0.5,0.7], 
              in_dim = 30, nbr_expe = 4,in_channels =30, hidden_channels =10, embedding_dim =30):


    datasets = [name + str(miss) for miss in miss_rate]
    datasets_and_std = []
    for d in datasets: 
        datasets_and_std.append(d)
        datasets_and_std.append(d + 'std')
    methods =['GAT', 'GCN', 'NODE']
    methods_wl_nwl = []
    for m in methods: 
        methods_wl_nwl.append(m + 'wl' )
        methods_wl_nwl.append(m + 'nwl')
    dict_results = {method : {das:0 for das in datasets_and_std} for method in methods_wl_nwl}
    
    
    # faut faire le changement ici sur y
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    dataset = Planetoid(root=data_dir, name=name)
    data = dataset[0]
    in_dim = 30
    d_F = [10,10,10]
    nbr_layers = len(d_F)+1
    in_channels, hidden_channels, embedding_dim = in_dim,10,in_dim
    num_node_initial = data.x.shape[0]
    initial_nodes = list(range(data.x.shape[0]))
    nbr_to_del = int(0*data.x.shape[0])
    to_suppr = random.sample(initial_nodes, nbr_to_del)
    to_keep = [i for i in range(num_node_initial) if i not in to_suppr]

    save_y = copy.deepcopy(data.y)
    save_y = save_y.numpy()
    num_classes = len(set(save_y))
    
    gat_gnn = GAT_GNN(in_channels, hidden_channels, embedding_dim)
    mlp_gat = MLP(embedding_dim=in_dim, hidden_dim=64, out_channels=num_classes)
    conv_gnn = CONV_GNN(in_channels, hidden_channels, embedding_dim)
    mlp_conv = MLP(embedding_dim=in_dim, hidden_dim=64, out_channels=num_classes)
    node_gnn = Node_GNN(d_F, nbr_layers)
    mlp_node = MLP(embedding_dim=in_dim, hidden_dim=64, out_channels=num_classes)

    X = data.x[to_keep, :] # We only work on saved nodes
    y = data.y[to_keep]  # We only work on saved nodes
    X = X.numpy()
    X = PCA(in_dim).fit_transform(X)
    X = torch.tensor(X).to(data.x.dtype)
    data.x = X
    data.y = y

    data.edge_index = new_edge_index(data, to_keep, to_suppr,num_node_initial)

    # ICI FAIRE L'EXPE 5 FOIS ok 
    accuracy_GCN_wl = list()
    accuracy_GAT_wl = list()
    accuracy_NODE_wl = list()
    accuracy_GCN_nwl = list()
    accuracy_GAT_nwl = list()
    accuracy_NODE_nwl = list()

    for _ in range(nbr_expe):
        train_mask, test_mask = split_train_test(data)
        optimizer = torch.optim.Adam(list(gat_gnn.parameters()) + list(mlp_gat.parameters()), lr=0.01)
        # Train the GNN + MLP model
        trained_gat, trained_mlp_gat, save_dict = train_gnn_mlp(data, train_mask, test_mask, gat_gnn, mlp_gat, optimizer)
        accuracy_GAT_wl.append(return_acc(data, trained_gat, trained_mlp_gat, test_mask))
        accuracy_GAT_nwl.append(return_acc(data, trained_gat, trained_mlp_gat, test_mask))
        
        optimizer = torch.optim.Adam(list(conv_gnn.parameters()) + list(mlp_conv.parameters()), lr=0.01)
        # Train the GNN + MLP model
        trained_conv, trained_mlp_conv, save_dict = train_gnn_mlp(data, train_mask, test_mask, conv_gnn, mlp_conv, optimizer)
        accuracy_GCN_wl.append(return_acc(data, trained_conv, trained_mlp_conv, test_mask))
        accuracy_GCN_nwl.append(return_acc(data, trained_conv, trained_mlp_conv, test_mask))

        optimizer = torch.optim.Adam(list(node_gnn.parameters()) + list(mlp_node.parameters()), lr=0.01)
        # Train the GNN + MLP model
        trained_gnn, trained_mlp, save_dict = train_gnn_mlp(data, train_mask, test_mask, node_gnn, mlp_node, optimizer,epochs= 20)
        accuracy_NODE_wl.append(return_acc(data, trained_gnn, trained_mlp, test_mask))
        accuracy_NODE_nwl.append(return_acc(data, trained_gnn, trained_mlp, test_mask))

    dict_results['GATwl'][name + str(0.0)] = np.mean(accuracy_GAT_wl)
    dict_results['GATwl'][name + str(0.0) +'std'] = np.std(accuracy_GAT_wl)
    dict_results['GATnwl'][name + str(0.0)] = np.mean(accuracy_GAT_nwl)
    dict_results['GATnwl'][name + str(0.0) +'std'] = np.std(accuracy_GAT_nwl)

    dict_results['GCNwl'][name + str(0.0)] = np.mean(accuracy_GCN_wl)
    dict_results['GCNwl'][name + str(0.0) +'std'] = np.std(accuracy_GCN_wl)
    dict_results['GCNnwl'][name + str(0.0)] = np.mean(accuracy_GCN_nwl)
    dict_results['GCNnwl'][name + str(0.0) +'std'] = np.std(accuracy_GCN_nwl)

    dict_results['NODEwl'][name + str(0.0)] = np.mean(accuracy_NODE_wl)
    dict_results['NODEwl'][name + str(0.0) +'std'] = np.std(accuracy_NODE_wl)
    dict_results['NODEnwl'][name + str(0.0)] = np.mean(accuracy_NODE_nwl)
    dict_results['NODEnwl'][name + str(0.0) +'std'] = np.std(accuracy_NODE_nwl)
    for miss in tqdm(miss_rate[1:]):
        accuracy_GCN_wl = list()
        accuracy_GAT_wl = list()
        accuracy_NODE_wl = list()
        accuracy_GCN_nwl = list()
        accuracy_GAT_nwl = list()
        accuracy_NODE_nwl = list()
        for i in range(nbr_expe):
            data = dataset[0]
            num_node_initial = data.x.shape[0]
            initial_nodes = list(range(data.x.shape[0]))
            nbr_to_del = int(miss*data.x.shape[0])
            to_suppr = random.sample(initial_nodes, nbr_to_del)
            to_keep = [i for i in range(num_node_initial) if i not in to_suppr]

            num_classes = len(set(data.y))
            mlp_gat = MLP(embedding_dim=in_dim, hidden_dim=64, out_channels=num_classes)
            mlp_conv = MLP(embedding_dim=in_dim, hidden_dim=64, out_channels=num_classes)
            mlp_node = MLP(embedding_dim=in_dim, hidden_dim=64, out_channels=num_classes)

            X = data.x[to_keep, :] # We only work on saved nodes
            y = data.y[to_keep]  # We only work on saved nodes
            X = X.numpy()
            X = PCA(in_dim).fit_transform(X)
            X = torch.tensor(X).to(data.x.dtype)
            data.x = X
            data.y = y

            data.edge_index = new_edge_index(data, to_keep, to_suppr,num_node_initial)

            train_mask, test_mask = split_train_test(data)

            optimizer = torch.optim.Adam(list(mlp_gat.parameters()), lr=0.01)
            # Train the GNN + MLP model
            _, trained_mlp_gat_expe, save_dict = train_gnn_mlp(data, train_mask, test_mask, trained_gat, mlp_gat, optimizer)
            accuracy_GAT_wl.append(return_acc(data, trained_gat, trained_mlp_gat_expe, test_mask))
            accuracy_GAT_nwl.append(return_acc(data, trained_gat, trained_mlp_gat, test_mask))

            optimizer = torch.optim.Adam(list(mlp_conv.parameters()), lr=0.01)
            # Train the GNN + MLP model
            _, trained_mlp_conv_expe, save_dict = train_gnn_mlp(data, train_mask, test_mask, trained_conv, mlp_conv, optimizer)
            accuracy_GCN_wl.append(return_acc(data, trained_conv, trained_mlp_conv_expe, test_mask))
            accuracy_GCN_nwl.append(return_acc(data, trained_conv, trained_mlp_conv, test_mask))

            optimizer = torch.optim.Adam(list(mlp_node.parameters()), lr=0.01)
            # Train the GNN + MLP model
            _, trained_mlp_node_expe, save_dict = train_gnn_mlp(data, train_mask, test_mask, trained_gnn, mlp_node, optimizer,epochs= 20)
            accuracy_NODE_wl.append(return_acc(data, trained_gnn, trained_mlp_node_expe, test_mask))
            accuracy_NODE_nwl.append(return_acc(data, trained_gnn, trained_mlp, test_mask))
        
        dict_results['GATwl'][name + str(miss)] = np.mean(accuracy_GAT_wl)
        dict_results['GATwl'][name + str(miss) +'std'] = np.std(accuracy_GAT_wl)
        dict_results['GATnwl'][name + str(miss)] = np.mean(accuracy_GAT_nwl)
        dict_results['GATnwl'][name + str(miss) +'std'] = np.std(accuracy_GAT_nwl)

        dict_results['GCNwl'][name + str(miss)] = np.mean(accuracy_GCN_wl)
        dict_results['GCNwl'][name + str(miss) +'std'] = np.std(accuracy_GCN_wl)
        dict_results['GCNnwl'][name + str(miss)] = np.mean(accuracy_GCN_nwl)
        dict_results['GCNnwl'][name + str(miss) +'std'] = np.std(accuracy_GCN_nwl)

        dict_results['NODEwl'][name + str(miss)] = np.mean(accuracy_NODE_wl)
        dict_results['NODEwl'][name + str(miss) +'std'] = np.std(accuracy_NODE_wl)
        dict_results['NODEnwl'][name + str(miss)] = np.mean(accuracy_NODE_nwl)
        dict_results['NODEnwl'][name + str(miss) +'std'] = np.std(accuracy_NODE_nwl)
        
    return dict_results