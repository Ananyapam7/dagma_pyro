import torch
import torch.nn as nn
import math
import numpy as np
import typing
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from tqdm.auto import tqdm
import copy


class LocallyConnected(nn.Module):
    """
    Implements a local linear layer, i.e. Conv1dLocal() with filter size 1.
    """
    
    def __init__(self, num_linear: int, input_features: int, output_features: int, bias: bool = True):
        r"""
        Parameters
        ----------
        num_linear : int
            Number of local linear layers.
        input_features : int
            Number of input features (m1).
        output_features : int
            Number of output features (m2).
        bias : bool, optional
            Whether to include bias or not. Default: ``True``.
        
        Attributes
        ----------
        weight : [d, m1, m2]
        bias : [d, m2]
        """
        super(LocallyConnected, self).__init__()
        self.num_linear = num_linear
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(num_linear,
                                                input_features,
                                                output_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_linear, output_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        k = 1.0 / self.input_features
        bound = math.sqrt(k)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""
        Implements the forward pass of the layer.

        Parameters
        ----------
        input : torch.Tensor
            Shape :math:`(n, d, m1)`

        Returns
        -------
        torch.Tensor
            Shape :math:`(n, d, m2)`
        """
        # [n, d, 1, m2] = [n, d, 1, m1] @ [1, d, m1, m2]
        out = torch.matmul(input.unsqueeze(dim=2), self.weight.unsqueeze(dim=0))
        out = out.squeeze(dim=2)
        if self.bias is not None:
            # [n, d, m2] += [d, m2]
            out += self.bias
        return out

    def extra_repr(self) -> str:
        """
        Returns a string with extra information from the layer.
        """
        return 'num_linear={}, in_features={}, out_features={}, bias={}'.format(
            self.num_linear, self.input_features, self.output_features,
            self.bias is not None
        )


class DagmaMLP(nn.Module): 
    """
    Class that models the structural equations for the causal graph using MLPs.
    """
    
    def __init__(self, dims: typing.List[int], bias: bool = True, dtype: torch.dtype = torch.double):
        r"""
        Parameters
        ----------
        dims : typing.List[int]
            Number of neurons in hidden layers of each MLP representing each structural equation.
        bias : bool, optional
            Flag whether to consider bias or not, by default ``True``
        dtype : torch.dtype, optional
            Float precision, by default ``torch.double``
        """
        torch.set_default_dtype(dtype)
        super(DagmaMLP, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        self.dims, self.d = dims, dims[0]
        self.I = torch.eye(self.d)
        self.fc1 = nn.Linear(self.d, self.d * dims[1], bias=bias)
        nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(self.d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Applies the current states of the structural equations to the dataset X

        Parameters
        ----------
        x : torch.Tensor
            Input dataset with shape :math:`(n,d)`.

        Returns
        -------
        torch.Tensor
            Result of applying the structural equations to the input data.
            Shape :math:`(n,d)`.
        """
        x = self.fc1(x)
        x = x.view(-1, self.dims[0], self.dims[1])
        for fc in self.fc2:
            x = torch.sigmoid(x)
            x = fc(x)
        x = x.squeeze(dim=2)
        return x

    def h_func(self, s: float = 1.0) -> torch.Tensor:
        r"""
        Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG

        Parameters
        ----------
        s : float, optional
            Controls the domain of M-matrices, by default 1.0

        Returns
        -------
        torch.Tensor
            A scalar value of the log-det acyclicity function :math:`h(\Theta)`.
        """
        fc1_weight = self.fc1.weight
        fc1_weight = fc1_weight.view(self.d, -1, self.d)
        A = torch.sum(fc1_weight ** 2, dim=1).t()  # [i, j]
        h = -torch.slogdet(s * self.I - A)[1] + self.d * np.log(s)
        return h

    def fc1_l1_reg(self) -> torch.Tensor:
        r"""
        Takes L1 norm of the weights in the first fully-connected layer

        Returns
        -------
        torch.Tensor
            A scalar value of the L1 norm of first FC layer. 
        """
        return torch.sum(torch.abs(self.fc1.weight))

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        r"""
        Computes the induced weighted adjacency matrix W from the first FC weights.
        Intuitively each edge weight :math:`(i,j)` is the *L2 norm of the functional influence of variable i to variable j*.

        Returns
        -------
        np.ndarray
            :math:`(d,d)` weighted adjacency matrix 
        """
        fc1_weight = self.fc1.weight
        fc1_weight = fc1_weight.view(self.d, -1, self.d)  
        A = torch.sum(fc1_weight ** 2, dim=1).t() 
        W = torch.sqrt(A)
        W = W.cpu().detach().numpy()  # [i, j]
        return W


class DagmaNonlinear:
    """
    Class that implements the DAGMA algorithm using Pyro's inference engine for non-linear models.
    """
    
    def __init__(self, model: nn.Module, verbose: bool = False, dtype: torch.dtype = torch.double) -> None:
        """
        Parameters
        ----------
        model : nn.Module
            Neural net that models the structural equations.
        verbose : bool, optional
            If true, the loss/score and h values will print to stdout every ``checkpoint`` iterations,
            as defined in :py:meth:`~dagma.nonlinear.DagmaNonlinear.fit`. Defaults to ``False``.
        dtype : torch.dtype, optional
            Float number precision, by default ``torch.double``.
        """
        self.vprint = print if verbose else lambda *a, **k: None
        self.model = model
        self.dtype = dtype

    def model_pyro(self, X, mu, lambda1, s):
        d = X.shape[1]
        # Sample weights for fc1
        fc1_weight = pyro.sample("fc1_weight", dist.Normal(torch.zeros_like(self.model.fc1.weight),
                                                           torch.ones_like(self.model.fc1.weight)).to_event(2))
        self.model.fc1.weight.data = fc1_weight

        # Sample weights for fc2
        for idx, layer in enumerate(self.model.fc2):
            weight = pyro.sample(f"fc2_weight_{idx}", dist.Normal(torch.zeros_like(layer.weight),
                                                                    torch.ones_like(layer.weight)).to_event(2))
            layer.weight.data = weight
            if layer.bias is not None:
                bias = pyro.sample(f"fc2_bias_{idx}", dist.Normal(torch.zeros_like(layer.bias),
                                                                    torch.ones_like(layer.bias)).to_event(1))
                layer.bias.data = bias

        X_hat = self.model(X)
        loss = 0.5 * d * torch.log(torch.mean((X_hat - X) ** 2))

        h = self.model.h_func(s)

        obj = mu * (loss + lambda1 * self.model.fc1_l1_reg()) + h
        pyro.factor("objective", -obj)  # Negative because we minimize

    def guide_pyro(self, X, mu, lambda1, s):
        # Guide for fc1
        fc1_loc = pyro.param("fc1_loc", torch.zeros_like(self.model.fc1.weight))
        fc1_scale = pyro.param("fc1_scale", torch.ones_like(self.model.fc1.weight),
                               constraint=dist.constraints.positive)
        pyro.sample("fc1_weight", dist.Normal(fc1_loc, fc1_scale).to_event(2))

        # Guides for fc2
        for idx, layer in enumerate(self.model.fc2):
            weight_loc = pyro.param(f"fc2_weight_loc_{idx}", torch.zeros_like(layer.weight))
            weight_scale = pyro.param(f"fc2_weight_scale_{idx}", torch.ones_like(layer.weight),
                                      constraint=dist.constraints.positive)
            pyro.sample(f"fc2_weight_{idx}", dist.Normal(weight_loc, weight_scale).to_event(2))
            if layer.bias is not None:
                bias_loc = pyro.param(f"fc2_bias_loc_{idx}", torch.zeros_like(layer.bias))
                bias_scale = pyro.param(f"fc2_bias_scale_{idx}", torch.ones_like(layer.bias),
                                        constraint=dist.constraints.positive)
                pyro.sample(f"fc2_bias_{idx}", dist.Normal(bias_loc, bias_scale).to_event(1))

    def fit(self, 
            X: typing.Union[torch.Tensor, np.ndarray],
            lambda1: float = .02, 
            lambda2: float = .005,  # Not used in Pyro version
            T: int = 4, 
            mu_init: float = .1, 
            mu_factor: float = .1, 
            s: typing.Union[typing.List[float], float] = 1.0,
            warm_iter: int = 50000, 
            max_iter: int = 80000, 
            lr: float = .0003, 
            w_threshold: float = 0.3, 
            checkpoint: int = 1000,
        ) -> np.ndarray:
        r"""
        Runs the DAGMA algorithm using Pyro's inference engine and fits the model to the dataset.

        Parameters
        ----------
        X : typing.Union[torch.Tensor, np.ndarray]
            :math:`(n,d)` dataset.
        lambda1 : float, optional
            Coefficient of the L1 penalty, by default .02.
        lambda2 : float, optional
            Coefficient of the L2 penalty, not used in this Pyro version.
        T : int, optional
            Number of DAGMA iterations, by default 4.
        mu_init : float, optional
            Initial value of :math:`\mu`, by default 0.1.
        mu_factor : float, optional
            Decay factor for :math:`\mu`, by default .1.
        s : typing.Union[typing.List[float], float], optional
            Controls the domain of M-matrices, by default 1.0.
        warm_iter : int, optional
            Number of iterations for SVI for :math:`t < T`, by default 50000.
        max_iter : int, optional
            Number of iterations for SVI for :math:`t = T`, by default 80000.
        lr : float, optional
            Learning rate, by default .0003.
        w_threshold : float, optional
            Removes edges with weight value less than the given threshold, by default 0.3.
        checkpoint : int, optional
            If ``verbose`` is ``True``, then prints to stdout every ``checkpoint`` iterations, by default 1000.

        Returns
        -------
        np.ndarray
            Estimated DAG from data.
        
        .. important::

            If the output of :py:meth:`~dagma.nonlinear.DagmaNonlinear.fit` is not a DAG, then the user should try larger values of ``T`` (e.g., 6, 7, or 8) 
            before raising an issue in github.
        """
        torch.set_default_dtype(self.dtype)
        if isinstance(X, np.ndarray):
            self.X = torch.from_numpy(X).type(self.dtype)
        elif isinstance(X, torch.Tensor):
            self.X = X.type(self.dtype)
        else:
            raise ValueError("X should be numpy array or torch Tensor.")
        
        self.checkpoint = checkpoint
        mu = mu_init
        if isinstance(s, list):
            if len(s) < T: 
                self.vprint(f"Length of s is {len(s)}, using last value in s for iteration t >= {len(s)}")
                s = s + [s[-1]] * (T - len(s))
        elif isinstance(s, (int, float)):
            s = [s] * T
        else:
            raise ValueError("s should be a list, int, or float.") 

        # Setup Pyro optimizer and SVI
        optimizer = Adam({"lr": lr})
        svi = SVI(self.model_pyro, self.guide_pyro, optimizer, loss=Trace_ELBO())

        with tqdm(total=(T-1)*warm_iter + max_iter) as pbar:
            for i in range(int(T)):
                self.vprint(f'\nDagma iter t={i+1} -- mu: {mu}', 30*'-')
                inner_iters = int(max_iter) if i == T - 1 else int(warm_iter)
                
                for iter in range(1, inner_iters + 1):
                    loss = svi.step(self.X, mu, lambda1, s[i])
                    if iter % checkpoint == 0 or iter == inner_iters:
                        self.vprint(f'Iteration {iter} - ELBO Loss: {loss}')
                    pbar.update(1)
                
                # Update mu
                mu *= mu_factor

        # Retrieve learned weights and compute adjacency matrix
        with torch.no_grad():
            W_est = self.model.fc1_to_adj()
            W_est[np.abs(W_est) < w_threshold] = 0
        return W_est


def test():
    from timeit import default_timer as timer
    import utils  # Ensure that utils is in the PYTHONPATH or adjust the import accordingly

    utils.set_random_seed(1)
    torch.manual_seed(1)
    
    

    n, d, s0, graph_type, sem_type = 1000, 20, 20, 'ER', 'mlp'
    
    B_true = utils.simulate_dag(d, s0, graph_type)
    
    X = utils.simulate_nonlinear_sem(B_true, n, sem_type)
    print("test")
    

    eq_model = DagmaMLP(dims=[d, 10, 1], bias=True)
    dagma = DagmaNonlinear(eq_model, verbose=True)
    start = timer()
    W_est = dagma.fit(X, lambda1=0.02)
    end = timer()
    acc = utils.count_accuracy(B_true, W_est != 0)
    print(acc)
    print(f'time: {end-start:.4f}s')


if __name__ == '__main__':
    test()