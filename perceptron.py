import numpy as np

class Perceptron:
    """
    Perceptron Algorithm Implementation
    
    Mathematical Foundation:
    - Decision Boundary: y = f(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)
    - Matrix form: y = f(w·x + b) where w·x is dot product
    - Weight Update: wᵢ = wᵢ + η(y_true - y_pred)xᵢ
    - Bias Update: b = b + η(y_true - y_pred)
    - Learning Rate: η (controls step size)
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Initialize Perceptron Parameters
        
        Args:
            learning_rate (η): How fast the algorithm learns (0 < η ≤ 1)
            n_iterations: Maximum number of training epochs
        """
        self.learning_rate = learning_rate  # η (eta) in mathematical notation
        self.n_iterations = n_iterations    # Maximum training epochs
        self.activation_func = unit_step_func  # f(x) = step function
        self.weights = None    # w = [w₁, w₂, ..., wₙ] - feature weights
        self.bias = None       # b = bias term (threshold)

    def fit(self, X, y):
        """
        Train the Perceptron using the Perceptron Learning Rule
        
        Mathematical Process:
        1. Initialize: w = 0, b = 0
        2. For each sample (xᵢ, yᵢ):
           - Calculate: ŷᵢ = f(w·xᵢ + b)  [prediction]
           - Update: w = w + η(yᵢ - ŷᵢ)xᵢ   [weight update rule]
           - Update: b = b + η(yᵢ - ŷᵢ)     [bias update rule]
        
        Args:
            X: Input features [n_samples × n_features]
            y: Target labels [n_samples]
        """
        # Step 1: Initialize weights and bias to zero
        # Mathematical: w = [0, 0, ..., 0], b = 0
        self.weights = np.zeros(X.shape[1])  # w ∈ ℝⁿ (n = number of features)
        self.bias = 0  # b ∈ ℝ (single bias term)

        # Step 2: Training loop - iterate through data multiple times
        for _ in range(self.n_iterations):
            for i, x in enumerate(X):
                # Step 3: Forward Pass - Calculate prediction
                # Mathematical: net_input = w·x + b = Σ(wⱼ × xⱼ) + b
                net_input = np.dot(x, self.weights) + self.bias
                
                # Step 4: Apply activation function
                # Mathematical: ŷ = f(net_input) = f(w·x + b)
                y_pred = self.activation_func(net_input)
                
                # Step 5: Calculate error and update weights/bias
                # Mathematical: error = (y_true - y_pred)
                error = y[i] - y_pred
                
                # Weight Update Rule: wⱼ = wⱼ + η × error × xⱼ
                # This moves the decision boundary towards correct classification
                update = self.learning_rate * error  # η × (yᵢ - ŷᵢ)
                self.weights += update * x           # w = w + η(yᵢ - ŷᵢ)xᵢ
                self.bias += update                  # b = b + η(yᵢ - ŷᵢ)

        return self

    def predict(self, X):
        """
        Make predictions on new data
        
        Mathematical Process:
        For each sample x:
        1. Calculate: net_input = w·x + b
        2. Apply activation: y = f(net_input)
        
        Args:
            X: Input features [n_samples × n_features]
            
        Returns:
            predictions: Binary predictions [n_samples]
        """
        # Apply learned decision boundary: y = f(w·x + b)
        return np.array([
            self.activation_func(np.dot(x, self.weights) + self.bias) 
            for x in X
        ])

def unit_step_func(x):
    """
    Unit Step Activation Function (Heaviside Step Function)
    
    Mathematical Definition:
    f(x) = {
        1  if x ≥ 0  (positive or zero)
        0  if x < 0   (negative)
    }
    
    This creates a hard decision boundary at x = 0
    - If w·x + b ≥ 0 → predict class 1
    - If w·x + b < 0 → predict class 0
    
    Args:
        x: Net input (w·x + b)
        
    Returns:
        1 or 0 (binary classification)
    """
    return 1 if x >= 0 else 0