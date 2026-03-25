import numpy as np
from utils import softmax

class MLP:
    def __init__(self, input_dim, hidden_dim, num_classes):
        """
        Args:
            input_dim: 64 (for digits)
            hidden_dim: 32 (number of hidden neurons)
            num_classes: 10 (digits 0-9)
        """
        self.d = input_dim
        self.h = hidden_dim
        self.k = num_classes
        
        # Xavier/He initialization (mentioned in PDF)
        # Xavier: multiply by sqrt(2 / (fan_in + fan_out))
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, num_classes) * np.sqrt(2.0 / (hidden_dim + num_classes))
        self.b2 = np.zeros(num_classes)
        
        # For storing intermediate values during forward pass
        self.cache = {}
    
    def forward(self, X):
        """
        Forward pass through the network.
        
        Args:
            X: (n, d) input data
        Returns:
            probs: (n, k) predicted probabilities
        """
        # Layer 1: X → z1 → h
        z1 = X @ self.W1 + self.b1  # (n, h)
        h = np.tanh(z1)              # (n, h) - activation function
        
        # Layer 2: h → z2 → probs
        z2 = h @ self.W2 + self.b2   # (n, k)
        probs = softmax(z2)          # (n, k)
        
        # Store for backward pass
        self.cache = {'X': X, 'z1': z1, 'h': h, 'z2': z2, 'probs': probs}
        
        return probs