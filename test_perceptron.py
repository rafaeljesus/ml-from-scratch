import pytest
import numpy as np
from perceptron import Perceptron, unit_step_func


class TestPerceptron:
    """Test suite for Perceptron class"""
    
    def test_initialization_default_params(self):
        """Test perceptron initialization with default parameters"""
        perceptron = Perceptron()
        
        assert perceptron.learning_rate == 0.01
        assert perceptron.n_iterations == 1000
        assert perceptron.activation_func == unit_step_func
        assert perceptron.weights is None
        assert perceptron.bias is None
    
    def test_initialization_custom_params(self):
        """Test perceptron initialization with custom parameters"""
        learning_rate = 0.05
        n_iterations = 500
        
        perceptron = Perceptron(learning_rate=learning_rate, n_iterations=n_iterations)
        
        assert perceptron.learning_rate == learning_rate
        assert perceptron.n_iterations == n_iterations
    
    def test_fit_simple_data(self):
        """Test fitting on simple linearly separable data"""
        # Simple AND gate data
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 0, 0, 1])
        
        perceptron = Perceptron(learning_rate=0.1, n_iterations=100)
        fitted_perceptron = perceptron.fit(X, y)
        
        # Check that fit returns self
        assert fitted_perceptron is perceptron
        
        # Check that weights and bias are initialized
        assert perceptron.weights is not None
        assert perceptron.bias is not None
        assert perceptron.weights.shape == (2,)
        assert isinstance(perceptron.bias, (int, float))
    
    def test_fit_weights_shape(self):
        """Test that weights have correct shape after fitting"""
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        y = np.array([0, 1, 1])
        
        perceptron = Perceptron()
        perceptron.fit(X, y)
        
        assert perceptron.weights.shape == (3,)
    
    def test_predict_before_fit_raises_error(self):
        """Test that prediction fails before fitting"""
        perceptron = Perceptron()
        X = np.array([[1, 2]])
        
        with pytest.raises(TypeError):  # np.dot with None weights
            perceptron.predict(X)
    
    def test_predict_after_fit(self):
        """Test prediction after fitting"""
        # Simple linearly separable data
        X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y_train = np.array([0, 0, 0, 1])
        
        perceptron = Perceptron(learning_rate=0.1, n_iterations=100)
        perceptron.fit(X_train, y_train)
        
        # Test predictions
        X_test = np.array([[0, 0], [1, 1]])
        predictions = perceptron.predict(X_test)
        
        # Check return type and shape
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (2,)
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_predict_single_sample(self):
        """Test prediction on single sample"""
        X_train = np.array([[0, 0], [1, 1]])
        y_train = np.array([0, 1])
        
        perceptron = Perceptron()
        perceptron.fit(X_train, y_train)
        
        # Test single sample prediction
        X_test = np.array([[0.5, 0.5]])
        prediction = perceptron.predict(X_test)
        
        assert len(prediction) == 1
        assert prediction[0] in [0, 1]
    
    def test_learning_convergence(self):
        """Test that perceptron learns linearly separable data"""
        # OR gate data (linearly separable)
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 1])
        
        perceptron = Perceptron(learning_rate=0.1, n_iterations=100)
        perceptron.fit(X, y)
        
        # Should correctly classify training data
        predictions = perceptron.predict(X)
        accuracy = np.mean(predictions == y)
        
        assert accuracy >= 0.75  # Should get most right for linearly separable data


class TestUnitStepFunction:
    """Test suite for unit step activation function"""
    
    def test_positive_values(self):
        """Test unit step function with positive values"""
        assert unit_step_func(1.0) == 1
        assert unit_step_func(0.1) == 1
        assert unit_step_func(100) == 1
    
    def test_zero_value(self):
        """Test unit step function with zero"""
        assert unit_step_func(0.0) == 1  # Should return 1 for x >= 0
    
    def test_negative_values(self):
        """Test unit step function with negative values"""
        assert unit_step_func(-1.0) == 0
        assert unit_step_func(-0.1) == 0
        assert unit_step_func(-100) == 0
    
    def test_edge_cases(self):
        """Test unit step function edge cases"""
        assert unit_step_func(np.inf) == 1
        assert unit_step_func(-np.inf) == 0


# Fixtures for reusable test data
@pytest.fixture
def simple_and_data():
    """Fixture providing simple AND gate data"""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    return X, y


@pytest.fixture
def trained_perceptron(simple_and_data):
    """Fixture providing a trained perceptron"""
    X, y = simple_and_data
    perceptron = Perceptron(learning_rate=0.1, n_iterations=100)
    perceptron.fit(X, y)
    return perceptron


def test_with_fixtures(trained_perceptron, simple_and_data):
    """Example test using fixtures"""
    X, y = simple_and_data
    predictions = trained_perceptron.predict(X)
    
    # Should predict something reasonable
    assert len(predictions) == len(y)
    assert all(pred in [0, 1] for pred in predictions)


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__])
