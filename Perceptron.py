import numpy as np
from numpy.typing import NDArray

class Perceptron:
    def __init__(self, inputs: NDArray[np.float64], weights: NDArray[np.float64], y_objectives: NDArray[np.float64]):
        self.inputs = inputs
        self.weights = weights
        self.y_objectives = y_objectives

    def calculate_scalar_product(self):
        # y_calculated = []

        # for i in range(self.inputs.shape[0]): 
        #     y = 0
        #     for j in range(self.inputs.shape[1]): 
        #         value = self.inputs[i, j]
        #         weight = self.weights[j]
        #         y += value * weight
        #     y_calculated.append(y)

        # return y_calculated

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
        # adjust_weights = []
        # x_t = self.inputs.T
        # for i in range(x_t.shape[0]):
        #     w = 0
        #     for j in range(x_t.shape[1]):
        #         value = x_t[i, j]
        #         err = errors[j]
        #         w += value * err
        #     adjust_weights.append(w)
        # return np.array([-(learning_rate * w) for w in adjust_weights]) 
        return -learning_rate * np.dot(self.inputs.T, errors)

       
    def update_weights(self, adjust_weights):
        # new_weights = []

        # for i in range(len(self.weights)):
        #     new_weights.append(self.weights[i] + adjust_weights[i])

        # return new_weights
        return self.weights + adjust_weights
            
                
              