from __future__ import print_function, division

import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """
    def __init__(self, node_fea_len, edge_fea_len):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        node_fea_len: int
          Number of node hidden features.
        edge_fea_len: int
          Number of edge features.
        """
        super(ConvLayer, self).__init__()
        self.node_fea_len = node_fea_len
        self.edge_fea_len = edge_fea_len
        # fully connected layer (pay attention to dimension)
        self.fc_full = nn.Linear(2*self.node_fea_len+self.edge_fea_len,
                                 self.node_fea_len)
        # activation function
        self.relu = nn.ReLU()

    def forward(self, nfeature, neighbor, efeature):
        # N is maximum number of nodes 
        # M is number of input node features
        # K is number of edge features
        _, N, M = nfeature.shape
        _, _, _, K = efeature.shape
        # calculate neighbor node features
        # the dimension of neighbor matrix should be N * (M + K)
        # sum the feature of neighboring nodes
        node_nbr_fea = torch.bmm(neighbor, nfeature)
        # sum the feature of edges
        edge_nbr_fea = torch.sum(efeature, dim = 2)
        # concatenate the two features
        total_nbr_fea = torch.cat((node_nbr_fea, edge_nbr_fea), dim = 2)
        # concatenate with the node feature itself
        total_fea = torch.cat((total_nbr_fea, nfeature), dim = 2)
        # perform convolution through fully connected layer
        total_gated_fea = self.fc_full(total_fea)
        # apply activation function
        out = self.relu(total_gated_fea)
        return out


class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """
    def __init__(self, orig_node_fea_len, edge_fea_len, max_node,
                 node_fea_len=64, n_conv=3, h_fea_len=128, n_h=1):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        orig_node_fea_len: int
          Number of node features in the input.
        edge_fea_len: int
          Number of edge features.
        node_fea_len: int
          Number of hidden node features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        """
        super(CrystalGraphConvNet, self).__init__()
        # firstly, conver the dimension of feature matrix from input value to the value in convolutional layer
        self.embedding = nn.Linear(orig_node_fea_len, node_fea_len)
        # several convolutional layers
        self.convs = nn.ModuleList([ConvLayer(node_fea_len=node_fea_len,
                                    edge_fea_len=edge_fea_len)
                                    for _ in range(n_conv)])
        # fully connected layer
        self.conv_to_fc = nn.Linear(node_fea_len, h_fea_len)
        # activation function
        self.conv_to_fc_softplus = nn.Softplus()
        # more hidden layers
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.relus = nn.ModuleList([nn.ReLU()
                                             for _ in range(n_h-1)])
        # fully connected layer for final input
        self.totalfc = nn.Sequential(
                # 3 here because there are three input conductivity values
                nn.Linear(max_node * h_fea_len, 1024),
                nn.ReLU(),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128,3)
        )

    def forward(self, nfeature, neighbor, efeature):
        # first fully connected layer (node-wise)
        nfeature = self.embedding(nfeature)
        # several convolutions (node-wise)
        for conv_func in self.convs:
            nfeature = conv_func(nfeature, neighbor, efeature)
        # fully connected layer (node-wise)
        nfeature = self.conv_to_fc(self.conv_to_fc_softplus(nfeature))
        # activation layer (node-wise)
        nfeature = self.conv_to_fc_softplus(nfeature)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, relu in zip(self.fcs, self.relus):
                nfeature = relu(fc(nfeature))
        # flatten the feature matrix (dimension now should be max_node * h_fea_len)
        nfeature = nfeature.view(nfeature.size()[0],-1)
        # go through fully connected layer
        out = self.totalfc(nfeature)
        return out


