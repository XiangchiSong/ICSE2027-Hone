import numpy as np
import torch
from torch.optim import Optimizer
from models import NumpyModel


class ServerOpt:
    """
    Server optimizer base class for use with AdaptiveFedOpt.
    """

    def apply_gradients(self, model, grads):
        """
        Return copy of updated model.

        Args:
            - model: (NumpyModel) global model before step
            - grads: (NumpyModel) round psuedogradient

        Returns:
            (NumpyModel) updated model
        """
        raise NotImplementedError()


class ServerAdam(ServerOpt):
    """
    FedAdam server optimiser.
    """

    def __init__(self, params, lr, beta1, beta2, epsilon):
        """
        Returns a new ServerAdam instance. Uses params argument to initialise
        1st and 2nd moment. Learning rate is fixed as per AdaptiveFedOpt paper,
        not uses the learning rate schedule from original Adam paper.

        Args:
            - params:   (NumpyModel) copy of client model parameters
            - lr:       (float)      learning rate
            - beta1:    (float)      1st moment estimate decay rate
            - beta2:    (float)      2nd moment estimate decay rate
            - epsilon:  (float)      stability parameter
        """
        self.m = params.zeros_like()
        self.v = params.zeros_like()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def apply_gradients(self, model, grads):
        """
        Return model with one step of Adam.

        Args:
            - model: (NumpyModel) global model before step
            - grads: (NumpyModel) round psuedogradient

        Returns:
            (NumpyModel) updated model
        """
        self.m = (self.beta1 * self.m) + (1 - self.beta1) * grads
        self.v = (self.beta2 * self.v) + (1 - self.beta2) * (grads**2)

        # uses constant learning rate as per AdaptiveFedOpt paper
        return model - (self.m * self.lr) / ((self.v**0.5) + self.epsilon)


class pFedMeOptimizer(Optimizer):
    """
    Optimizer to use for pFedMe simulations.
    """

    def __init__(
        self, params, device, pfedme_lr=0.01, pfedme_lambda=0.1, pfedme_mu=0.001
    ):
        """
        Return a new pFedMe optimizer. The passed mu parameter does not
        explicitly feature in the pFedMe Algorithm, is used for weight decay.

        Args:
            - params: (iterable)        of nn.Module parameters
            - device: (torch.device)    where to place optimizer
            - pfedme_lr:     (float)           learning rate
            - pfedme_lamda:  (float)           pFedMe lambda parameter
            - pfedme_mu:     (float)           pFedMe mu parameter
        """
        if pfedme_lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(pfedme_lr))

        defaults = dict(
            pfedme_lr=pfedme_lr, pfedme_lambda=pfedme_lambda, pfedme_mu=pfedme_mu
        )
        super(pFedMeOptimizer, self).__init__(params, defaults)
        self.device = device

    def step(self, omega, closure=None):
        """
        One step of pFedMe.

        Args:
            - omega: (NumpyModel) local model, omega is the list of [localweight]
            - closure: (callable) function to compute loss
        """
        loss = None
        if closure is not None:
            loss = closure

        # apply pFedMe update rule
        for group in self.param_groups:
            # p means the weight parameter in the current optimizer object
            for p, localweight in zip(group["params"], omega):

                if p.grad is None:
                    continue

                w = torch.tensor(localweight).to(self.device)
                p.data = p.data - group["pfedme_lr"] * (
                    p.grad.data
                    + group["pfedme_lambda"] * (p.data - w)
                    + group["pfedme_mu"] * p.data
                )

        return group["params"], loss


class ClientOpt:
    """
    Client optimiser base class for use with FedAvg/AdaptiveFedOpt.
    """

    def get_params(self):
        """
        Returns:
            (NumpyModel) copy of all optimiser parameters.
        """
        raise NotImplementedError()

    def set_params(self, params):
        """
        Set all optimiser parameters.

        Args:
            - params: (NumpyModel) values to set
        """
        raise NotImplementedError()

    def get_bn_params(self, setting=0):
        """
        Return only BN parameters. Setting can be one of the following
        {0: usyb, 1: yb, 2: us, 3: none} to get different types of parameters.

        Args:
            - setting (int) param types to get

        Returns:
            list of numpy.ndarrays
        """
        raise NotImplementedError()

    def set_bn_params(self, params, setting=0):
        """
        Set only BN parameters. Setting can be one of the following
        {0: usyb, 1: yb, 2: us, 3: none} to get different types of parameters.

        Args:
            - params  (list) of numpy.ndarray values to set
            - setting (int) param types to get
        """
        raise NotImplementedError()


class ClientSGD(torch.optim.SGD, ClientOpt):
    """
    Client SGD optimizer for FedAvg and AdaptiveFedOpt.
    """

    def __init__(self, params, lr, weight_decay=0.0):
        super(ClientSGD, self).__init__(params, lr, weight_decay=weight_decay)

    def get_params(self):
        """
        Returns:
            (NumpyModel) copy of all optimiser parameters.
        """
        return NumpyModel([])

    def set_params(self, params):
        """
        Set all optimiser parameters.

        Args:
            - params: (NumpyModel) values to set
        """
        pass

    def get_bn_params(self, model, setting=0):
        """
        Vanilla SGD has no optimisation parameters. Returns empty list.

        Returns:
            [] empty list.
        """
        return []

    def set_bn_params(self, params, model, setting=0):
        """
        Vanilla SGD has no optimisation parameters. Does nothing.
        """
        pass

    def step(self, closure=None, beta=None):
        """
        SGD step.

        Args:
            - beta: (float) optional different learning rate.
        """
        loss = None
        if closure is not None:
            loss = closure

        # apply SGD update rule
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if beta is None:
                    p.data.add_(d_p, alpha=-group["lr"])
                else:
                    p.data.add_(d_p, alpha=-beta)

        return loss


class ClientDittoOptimizer(torch.optim.SGD, ClientOpt):
    """
    Client optimizer for Ditto algorithm that handles personalized model updates.
    Implements the gradient modification: ∇F_k(v_k) + λ(v_k - w)
    """

    def __init__(self, params, lr, ditto_lambda, weight_decay=0.0):
        """
        Initialize Ditto optimizer.

        Args:
            params: Model parameters to optimize
            lr: Learning rate
            ditto_lambda: Ditto regularization parameter λ
            weight_decay: L2 weight decay
        """
        super(ClientDittoOptimizer, self).__init__(
            params, lr, weight_decay=weight_decay
        )
        self.ditto_lambda = ditto_lambda

    def get_params(self):
        """
        Returns:
            (NumpyModel) copy of all optimiser parameters.
        """
        return NumpyModel([])  # SGD has no internal state to track

    def set_params(self, params):
        """
        Set all optimiser parameters.

        Args:
            - params: (NumpyModel) values to set
        """
        pass  # SGD has no internal state to set

    def get_bn_params(self, model, setting=0):
        """
        Vanilla SGD has no optimisation parameters. Returns empty list.

        Returns:
            [] empty list.
        """
        return []

    def set_bn_params(self, params, model, setting=0):
        """
        Vanilla SGD has no optimisation parameters. Does nothing.
        """
        pass

    def step(self, global_model_params=None, closure=None):
        """
        Performs a single optimization step with Ditto gradient modification.

        Args:
            global_model_params: (NumpyModel, optional) parameters of the global model w
                                If None, performs standard SGD without Ditto regularization
            closure: (callable, optional) A closure that reevaluates the model

        Returns:
            loss value if closure is provided
        """
        loss = None
        if closure is not None:
            loss = closure()

        # If no global model params provided, perform standard SGD
        if global_model_params is None:
            return super(ClientDittoOptimizer, self).step(closure)

        # Convert global model params to list of tensors for easier access
        # Get device from first parameter
        device = self.param_groups[0]["params"][0].device
        global_params_list = []
        for param in global_model_params.params:
            global_params_list.append(torch.tensor(param, device=device))

        # Apply Ditto gradient modification

        global_param_idx = 0
        for group in self.param_groups:
            for p in group["params"]:

                if global_param_idx >= len(global_params_list):
                    break

                global_param_tensor = global_params_list[global_param_idx]

                if p.grad is not None:

                    if p.data.shape != global_param_tensor.shape:

                        try:
                            global_param_tensor = global_param_tensor.view(p.data.shape)
                        except:
                            raise RuntimeError(
                                f"Shape mismatch at parameter {global_param_idx}: "
                                f"local param shape {p.data.shape} vs global param shape {global_param_tensor.shape}"
                            )

                    # Add proximal term to gradient: λ(v_k - w)
                    proximal_term_grad = self.ditto_lambda * (
                        p.data - global_param_tensor
                    )
                    p.grad.data.add_(proximal_term_grad)

                    # Apply standard SGD update with modified gradient
                    d_p = p.grad.data
                    if group["weight_decay"] != 0:
                        d_p.add_(p.data, alpha=group["weight_decay"])

                    p.data.add_(d_p, alpha=-group["lr"])

                global_param_idx += 1

        return loss


class ClientSGD_perfedavg(torch.optim.SGD, ClientOpt):
    """
    Client SGD optimizer specifically for Per-FedAvg algorithm.
    This optimizer implements the correct meta-learning update for Per-FedAvg.
    """

    def __init__(self, params, lr, weight_decay=0.0):
        super(ClientSGD_perfedavg, self).__init__(params, lr, weight_decay=weight_decay)
        self.initial_params = None

    def get_params(self):
        """
        Returns:
            (NumpyModel) copy of all optimiser parameters.
        """
        return NumpyModel([])

    def set_params(self, params):
        """
        Set all optimiser parameters.

        Args:
            - params: (NumpyModel) values to set
        """
        pass

    def get_bn_params(self, model, setting=0):
        """
        Vanilla SGD has no optimisation parameters. Returns empty list.

        Returns:
            [] empty list.
        """
        return []

    def set_bn_params(self, params, model, setting=0):
        """
        Vanilla SGD has no optimisation parameters. Does nothing.
        """
        pass

    def set_initial_params(self, model_params):
        """
        Store the initial model parameters for Per-FedAvg meta-learning.

        Args:
            - model_params: Initial model parameters before local training
        """
        self.initial_params = [
            p.clone() if isinstance(p, torch.Tensor) else torch.tensor(p.copy())
            for p in model_params
        ]

    def step(self, closure=None, beta=None):
        """
        Per-FedAvg step with meta-learning update.

        Args:
            - beta: (float) Per-FedAvg meta-learning rate
        """
        loss = None
        if closure is not None:
            loss = closure

        if beta is not None and self.initial_params is not None:
            # Per-FedAvg meta-learning update
            # θ_new = θ_init + β * (θ_current - θ_init)
            # where θ_current has the gradients from the meta-gradient computation
            param_idx = 0
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        param_idx += 1
                        continue

                    # Get initial parameter value
                    initial_p = self.initial_params[param_idx]

                    # Compute the update direction based on the gradient
                    # The gradient represents the direction from meta-gradient computation
                    # We apply: θ_new = θ_init - β * gradient
                    p.data = initial_p.data - beta * p.grad.data

                    param_idx += 1
        else:
            # Standard SGD update
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    d_p = p.grad.data
                    p.data.add_(d_p, alpha=-group["lr"])

        return loss


class ClientAdam(torch.optim.Adam, ClientOpt):
    """
    Client Adam optimizer for FedAvg.
    """

    def __init__(
        self,
        params,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-07,
        weight_decay=0,
        amsgrad=False,
    ):
        """
        Returns a new ClientAdam.

        Args:
            - params:      (NumpyModel) copy of client model parameters
            - lr:          (float)      learning rate
            - betas:       (tuple)      two floats, 1st/2nd moment decay rates
            - eps:         (float)      stability parameter
            - weight_decay (float)      L2 decay rate
            - amsgrad      (bool)       whether to use the AMSGrad variant
        """
        super(ClientAdam, self).__init__(params, lr, betas, eps, weight_decay, amsgrad)

    def get_bn_params(self, model, setting=0):
        """
        Return only BN parameters. Setting can be one of the following
        {0: usyb, 1: yb, 2: us, 3: none} to get different types of parameters.

        Args:
            - setting (int) param types to get

        Returns:
            list of numpy.ndarrays
        """

        if setting in [2, 3]:
            return []

        # order is (weight m, weight v, bias m, bias v)
        params = []
        for bn in model.bn_layers:
            weight = self.state[bn.weight]
            bias = self.state[bn.bias]
            params.append(np.copy(weight["exp_avg"].cpu().numpy()))
            params.append(np.copy(weight["exp_avg_sq"].cpu().numpy()))
            params.append(np.copy(bias["exp_avg"].cpu().numpy()))
            params.append(np.copy(bias["exp_avg_sq"].cpu().numpy()))

        return params

    def set_bn_params(self, params, model, setting=0):
        """
        Set only BN parameters. Setting can be one of the following
        {0: usyb, 1: yb, 2: us, 3: none} to get different types of parameters.
        Order of parameters should be (weight m, weight v, bias m, bias v).
        Length of params argument will then be 4*num_bn_layers.

        Args:
            - params  (list) of numpy.ndarray values to set
            - setting (int) param types to get
        """
        if setting in [2, 3]:
            return

        i = 0
        for bn in model.bn_layers:
            weight = self.state[bn.weight]
            bias = self.state[bn.bias]
            weight["exp_avg"].copy_(torch.tensor(params[i]))
            weight["exp_avg_sq"].copy_(torch.tensor(params[i + 1]))
            bias["exp_avg"].copy_(torch.tensor(params[i + 2]))
            bias["exp_avg_sq"].copy_(torch.tensor(params[i + 3]))
            i += 4

    def get_params(self):
        """
        Order of values in returned NumpModel is (step_num, m, v), for each
        model parameter.

        Returns:
            (NumpyModel) copy of all optimiser parameters.
        """
        params = []
        for key in self.state.keys():
            params.append(self.state[key]["step"])
            params.append(self.state[key]["exp_avg"].cpu().numpy())
            params.append(self.state[key]["exp_avg_sq"].cpu().numpy())

        return NumpyModel(params)

    def set_params(self, params):
        """
        Order of values in params arg should be (step_num, m, v), for each
        model parameter.

        Args:
            (NumpyModel) parameters to set.
        """
        i = 0
        for key in self.state.keys():
            self.state[key]["step"] = params[i]
            self.state[key]["exp_avg"].copy_(torch.tensor(params[i + 1]))
            self.state[key]["exp_avg_sq"].copy_(torch.tensor(params[i + 2]))
            i += 3
