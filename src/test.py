import torch
import sys
from dagma import utils
from dagma.linear import DagmaLinear
from dagma.nonlinear import DagmaMLP, DagmaNonlinear

utils.set_random_seed(1)
# Create an Erdos-Renyi DAG of 20 nodes and 20 edges in expectation with Gaussian noise
# number of samples n = 500
n, d, s0 = 500, 20, 20 
graph_type, sem_type = 'ER', 'gauss'

B_true = utils.simulate_dag(d, s0, graph_type)
W_true = utils.simulate_parameter(B_true)
X = utils.simulate_linear_sem(W_true, n, sem_type)

model = DagmaLinear(loss_type='l2') # create a linear model with least squares loss
W_est = model.fit(X, lambda1=0.02) # fit the model with L1 reg. (coeff. 0.02)
acc = utils.count_accuracy(B_true, W_est != 0) # compute metrics of estimated adjacency matrix W_est with ground-truth
print(acc)