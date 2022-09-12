# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 23:39:23 2022

@author: daimi
"""

import numpy as np
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# get data
feature = np.loadtxt('feature_20.txt', skiprows = 1)
pos = feature[:,:3]
neighbor = np.loadtxt('neighbor_20.txt')

# get edge lists
edgelist = np.transpose(np.nonzero(neighbor))
edgelist = np.sort(edgelist, axis = 1)
edgelist = np.unique(edgelist, axis = 0)
numneighbor = np.sum(neighbor, axis = 1)

# set up graph
# link: https://networkx.org/documentation/stable/tutorial.html
nnode, nedge = np.shape(feature)[0], np.shape(edgelist)[0]

G = nx.Graph()
G.add_nodes_from([0,nnode-1])

for i in range(nedge):
    e = (edgelist[i,0], edgelist[i,1])
    G.add_edge(*e)

# 3D graph visualization
# link: https://networkx.org/documentation/stable/auto_examples/3d_drawing/plot_basic.html
node_xyz = np.array([pos[v, :] for v in sorted(G)])
edge_xyz = np.array([(pos[u, :], pos[v, :]) for u, v in G.edges()])

# Create the 3D figure
fig = plt.figure(dpi = 1200)
ax = fig.add_subplot(111, projection="3d")

# Plot the nodes - alpha is scaled by "depth" automatically
sc = ax.scatter(*node_xyz.T, s=80, c = numneighbor, alpha = 1, ec = 'w')
plt.colorbar(sc)
#ax.scatter(*node_xyz.T, s=80, c = '#005A87', alpha = 1, ec = 'w')

# Plot the edges
for vizedge in edge_xyz:
    ax.plot(*vizedge.T, c = '#000043', lw = 0.5, alpha = 0.8)


def _format_axes(ax):
    """Visualization options for the 3D axes."""
    # Turn gridlines off
    ax.grid(False)
    ax.axis('off')



_format_axes(ax)
fig.tight_layout()
plt.show()