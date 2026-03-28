import numpy as np

class SGD:
    """
    Update rule:
        theta = theta - lr * gradient

    Args:
        lr: Learning rate 
    """

    def __init__(self, lr=0.05):
        self.lr = lr

    def step(self, params, grads):
        """
        Perform one optimization step.

        Args:
            params: dict of parameter name -> numpy array
            grads: dict of parameter name -> gradient array
        """
        for key in params:
            params[key] = params[key] - self.lr * grads[key]


class Momentum:
    """
    SGD with Momentum optimizer.
    Momentum accumulates gradients exponentially to accelerate
    convergence and dampen oscillations.

    Update rule:
        v = beta * v + gradient
        theta = theta - lr * v

    Args:
        lr: Learning rate 
        beta: Momentum coefficient 
    """

    def __init__(self, lr=0.05, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v = {}  # velocity for each parameter

    def step(self, params, grads):
        """
        Perform one optimization step.

        Args:
            params: dict of parameter name -> numpy array
            grads: dict of parameter name -> gradient array
        """
        for key in params:
            # Initialize velocity if first step
            if key not in self.v:
                self.v[key] = np.zeros_like(params[key])

            # Update velocity: v = beta * v + gradient
            self.v[key] = self.beta * self.v[key] + grads[key]

            # Update parameters: theta = theta - lr * v
            params[key] = params[key] - self.lr * self.v[key]


class Adam:
    """
    Adam combines momentum with RMSprop-like
    adaptive learning rates, with bias correction.

    Update rule:
        m = beta1 * m + (1 - beta1) * gradient        
        v = beta2 * v + (1 - beta2) * gradient^2      
        m_hat = m / (1 - beta1^t)                     
        v_hat = v / (1 - beta2^t)                     
        theta = theta - lr * m_hat / (sqrt(v_hat) + eps)

    Args:
        lr: Learning rate 
        beta1: First moment decay rate 
        beta2: Second moment decay rate 
        eps: Small constant for numerical stability 
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}  # first moment (mean of gradients)
        self.v = {}  # second moment (mean of squared gradients)
        self.t = 0  # timestep for bias correction

    def step(self, params, grads):
        """
        Perform one optimization step.

        Args:
            params: dict of parameter name -> numpy array
            grads: dict of parameter name -> gradient array
        """
        self.t += 1

        for key in params:
            # Initialize moments if first step
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

            # Update first moment: m = beta1 * m + (1 - beta1) * g
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]

            # Update second moment: v = beta2 * v + (1 - beta2) * g^2
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (
                grads[key] ** 2
            )

            # Bias correction
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)

            # Update parameters
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)



