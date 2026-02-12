import numpy as np
from numpy.typing import NDArray

class Perceptron:
    def __init__(self, inputs: NDArray[np.float64], weights: NDArray[np.float64], y_objectives: NDArray[np.float64]):
        self.inputs = self._add_bias(inputs)
        self.weights = weights
        self.y_objectives = y_objectives
    
    def _add_bias(self, x):
        bias = np.ones((x.shape[0], 1))
        return np.hstack((bias, x))

    def calculate_scalar_product(self):
        return np.dot(self.inputs, self.weights)

    def activation(self, y_calculated: np.float64):
        return 1 if y_calculated >= 0 else 0
    
    def loss(self, y_calculated: NDArray[np.float64]):
        errors = []

        for i in range (len(self.y_objectives)):
            e = y_calculated[i] - self.y_objectives[i]
            errors.append(e)

        return errors

    def adjust_weights(self, errors: list[np.float64], learning_rate: float):
        return -learning_rate * np.dot(self.inputs.T, errors)
  
    def update_weights(self, adjust_weights):
        return self.weights + adjust_weights
            
                
              