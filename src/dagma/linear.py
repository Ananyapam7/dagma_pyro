import numpy as np
import scipy.linalg as sla
import numpy.linalg as la
from scipy.special import expit as sigmoid
from tqdm.auto import tqdm
import typing
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

__all__ = ["DagmaLinear"]

class DagmaLinear:
    """
    A Python object that contains the implementation of DAGMA for linear models using NumPy, SciPy, and Pyro.
    """

    def __init__(self, loss_type: str, verbose: bool = False, dtype: type = np.float64) -> None:
        r"""
        Parameters
        ----------
        loss_type : str
            One of ["l2", "logistic"]. ``l2`` refers to the least squares loss, while ``logistic``
            refers to the logistic loss. For continuous data: use ``l2``. For discrete 0/1 data: use ``logistic``.
        verbose : bool, optional
            If true, the loss/score and h values will print to stdout every ``checkpoint`` iterations,
            as defined in :py:meth:`~dagma.linear.DagmaLinear.fit`. Defaults to ``False``.
        dtype : type, optional
           Defines the float precision, for large number of nodes it is recommended to use ``np.float64``. 
           Defaults to ``np.float64``.
        """
        super().__init__()
        losses = ['l2', 'logistic']
        assert loss_type in losses, f"loss_type should be one of {losses}"
        self.loss_type = loss_type
        self.dtype = dtype
        self.vprint = print if verbose else lambda *a, **k: None

    def model(self, X, mu, lambda1, s):
        d = X.shape[1]
        W = pyro.sample("W", dist.Normal(torch.zeros(d, d), torch.ones(d, d)).to_event(2))
        
        # Score Function
        if self.loss_type == 'l2':
            loss = 0.5 * torch.trace((torch.eye(d) - W).T @ (self.cov @ (torch.eye(d) - W)))
        elif self.loss_type == 'logistic':
            R = X @ W
            loss = (torch.log1p(torch.exp(R)) - X * R).mean()
        
        # Log-Determinant Acyclicity Constraint
        M = s * torch.eye(d) - W * W
        sign, logdet = torch.slogdet(M)
        h = -logdet + d * torch.log(torch.tensor(s))
        
        # Objective
        obj = mu * (loss + lambda1 * torch.norm(W, p=1)) + h
        pyro.factor("objective", -obj)  # Negative because we minimize

    def guide(self, X, mu, lambda1, s):
        d = X.shape[1]
        W_loc = pyro.param("W_loc", torch.zeros(d, d))
        W_scale = pyro.param("W_scale", torch.ones(d, d), constraint=torch.distributions.constraints.positive)
        pyro.sample("W", dist.Normal(W_loc, W_scale).to_event(2))

    def fit(self, 
            X: np.ndarray,
            lambda1: float = 0.03, 
            w_threshold: float = 0.3, 
            T: int = 5,
            mu_init: float = 1.0, 
            mu_factor: float = 0.1, 
            s: typing.Union[typing.List[float], float] = [1.0, .9, .8, .7, .6], 
            warm_iter: int = 30000, 
            max_iter: int = 60000, 
            lr: float = 0.0003, 
            checkpoint: int = 1000, 
            beta_1: float = 0.99, 
            beta_2: float = 0.999,
            exclude_edges: typing.Optional[typing.List[typing.Tuple[int, int]]] = None, 
            include_edges: typing.Optional[typing.List[typing.Tuple[int, int]]] = None,
        ) -> np.ndarray :
        r"""
        Runs the DAGMA algorithm using Pyro's inference engine and returns a weighted adjacency matrix.
        
        [Documentation remains the same as original]
        """ 
        ## INITIALIZING VARIABLES 
        self.X, self.lambda1, self.checkpoint = X, lambda1, checkpoint
        self.n, self.d = X.shape
        self.Id = np.eye(self.d).astype(self.dtype)
        
        if self.loss_type == 'l2':
            self.X -= X.mean(axis=0, keepdims=True)
        
        self.exc_r, self.exc_c = None, None
        self.inc_r, self.inc_c = None, None
        
        if exclude_edges is not None:
            if (isinstance(exclude_edges, tuple) and 
                isinstance(exclude_edges[0], tuple) and 
                np.all(np.array([len(e) for e in exclude_edges]) == 2)):
                self.exc_r, self.exc_c = zip(*exclude_edges)
            else:
                raise ValueError("exclude_edges should be a tuple of edges, e.g., ((1,2), (2,3))")
        
        if include_edges is not None:
            if (isinstance(include_edges, tuple) and 
                isinstance(include_edges[0], tuple) and 
                np.all(np.array([len(e) for e in include_edges]) == 2)):
                self.inc_r, self.inc_c = zip(*include_edges)
            else:
                raise ValueError("include_edges should be a tuple of edges, e.g., ((1,2), (2,3))")        
            
        self.cov = X.T @ X / float(self.n)    
        self.W_est = np.zeros((self.d,self.d)).astype(self.dtype) # init W0 at zero matrix
        mu = mu_init
        if isinstance(s, list):
            if len(s) < T: 
                self.vprint(f"Length of s is {len(s)}, using last value in s for iteration t >= {len(s)}")
                s = s + [s[-1]] * (T - len(s))
        elif isinstance(s, (int, float)):
            s = [s] * T
        else:
            raise ValueError("s should be a list, int, or float.")    
        
        # Convert X to torch tensor
        X_torch = torch.tensor(X, dtype=torch.float32)
        self.cov = torch.tensor(self.cov, dtype=torch.float32)
        
        ## Setup Pyro Optimizer and SVI
        pyro.clear_param_store()
        optimizer = Adam({"lr": lr, "betas": (beta_1, beta_2)})
        svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())
        
        ## START DAGMA
        with tqdm(total=(T-1)*warm_iter + max_iter) as pbar:
            for i in range(int(T)):
                self.vprint(f'\nIteration -- {i+1}:')
                lr_adam, success = lr, False
                inner_iters = int(max_iter) if i == T - 1 else int(warm_iter)
                
                for iter in range(1, inner_iters + 1):
                    loss = svi.step(X_torch, mu, lambda1, s[i])
                    if iter % checkpoint == 0 or iter == inner_iters:
                        self.vprint(f'Iteration {iter} - ELBO Loss: {loss}')
                    pbar.update(1)
                
                # Update mu
                mu *= mu_factor
        
        # Retrieve learned W
        W_learned = pyro.param("W_loc").detach().numpy()
        self.W_est = W_learned
        self.W_est[np.abs(self.W_est) < w_threshold] = 0
        return self.W_est

def test():
    import sys
    sys.path.append('..')
    from dagma import utils
    from timeit import default_timer as timer
    utils.set_random_seed(1)
    
    n, d, s0 = 500, 20, 20 # the ground truth is a DAG of 20 nodes and 20 edges in expectation
    graph_type, sem_type = 'ER', 'gauss'
    
    B_true = utils.simulate_dag(d, s0, graph_type)
    W_true = utils.simulate_parameter(B_true)
    X = utils.simulate_linear_sem(W_true, n, sem_type)
    
    model = DagmaLinear(loss_type='l2', verbose=True)
    start = timer()
    W_est = model.fit(X, lambda1=0.02)
    end = timer()
    acc = utils.count_accuracy(B_true, W_est != 0)
    print(acc)
    print(f'time: {end-start:.4f}s')

if __name__ == '__main__':
    test()