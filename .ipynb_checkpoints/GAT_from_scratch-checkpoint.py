# Implementing  GAT using numpy
from pprint import pprint

import numpy as np

np.random.seed(10)

# Adjacency matrix
A = np.array([
    [1, 1, 1, 1],
    [1, 1, 0, 0],
    [1, 0, 1, 1],
    [1, 0, 1, 1],
])

# Generate random matrix of node features
X = np.random.uniform(-1, 1, (4, 4))

# Regular weight matrix
# (no_of_hidden_dims, no_of_nodes)
W = np.random.uniform(-1, 1, (2, 4))

# Attention matrix
# size: (1, dim_h x 2)
W_att = np.random.uniform(-1, 1, (1, 4))

# Connections from source nodes to destination nodes
connections = np.where(A > 0)

# Concatenate hidden vectors of source and destination nodes
# Then apply linear transformation: W_att
a = W_att @ np.concatenate([(X @ W.T)[connections[0]], (X @ W.T)[connections[1]]], axis=1).T


# Applying Leaky ReLU to the previous result
def leaky_relu(x, alpha=0.2):
    return np.maximum(alpha * x, x)


e = leaky_relu(a)

# Place these values in a matrix: shape = A.shape
# Means it should look like a adjacency matrix
E = np.zeros(A.shape)  # (no_of_nodes, no_of_nodes)
E[connections[0], connections[1]] = e[0]
print('New Attention Based Adj')
pprint(E)


# Normalize every row of attention scores.
def softmax2D(x, axis):
    e = np.exp(x - np.expand_dims(np.max(x, axis=axis), axis))
    sum = np.expand_dims(np.sum(e, axis=axis), axis)
    return e / sum


# Attention weights
W_alpha = softmax2D(E, 1)

# Calculate new matrix of embeddings H
H = A.T @ W_alpha @ X @ W.T
print('Hidden embeddings:')
pprint(H)
