from typing import Literal

from .matrix import *
from .activations import *

class NeuralNetwork:
    ''' Neural Network class '''
    
    def __init__(
        self,  
        hidden_nodes: list[int],
        activation: Literal['identity', 'sigmoid', 'tanh', 'relu'] = 'sigmoid',
        output_activation: Literal['identity', 'sigmoid', 'softmax'] = 'sigmoid',
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        batch_size: int = 200, 
        max_iter: int = 200,
        max_no_change_count: int = 10,
        tol: float = 1e-4,
        verbose: bool = False
    ):
        ''' Create a neural network with the given number of input, hidden and output nodes '''
        
        self.input_nodes: int = 0
        self.output_nodes: int = 0
        
        self.hidden_nodes = hidden_nodes
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.max_iter = max_iter
        self.max_no_change_count = max_no_change_count
        self.tol = tol
        self.verbose = verbose
        
        self.activation = ACTIVATIONS[activation]
        self.output_activation = ACTIVATIONS[output_activation]
        
        self.derivative = DERIVATIVES[activation]
        self.output_derivative = DERIVATIVES[output_activation]
        
        self.biases: list[Matrix] = []
        self.weights: list[Matrix] = []

        self.biases_update: list[Matrix] = []
        self.weights_update: list[Matrix] = []
                
    def initialize(self, input_nodes: int, output_nodes: int) -> None:
        ''' Initialize the weights and biases of the neural network '''
        
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        
        nodes = [input_nodes] + self.hidden_nodes + [output_nodes]
        
        self.biases.clear()
        self.weights.clear()
        
        for n_in, n_out in zip(nodes[:-1], nodes[1:]):
            bound = (1 / n_in) ** 0.5
            
            self.biases.append(Matrix(n_out, 1).randomize(bound))
            self.weights.append(Matrix(n_out, n_in).randomize(bound))
        
        self.biases_update.clear()
        self.weights_update.clear()
        
        self.biases_update.extend([matrix.zeros() for matrix in self.biases])
        self.weights_update.extend([matrix.zeros() for matrix in self.weights])
    
    def predict(self, X: list[list[float]]) -> list[list[float]]:
        ''' Predict the output of the neural network '''
        
        if len(self.weights) == 0:
            raise ValueError('The neural network must be fitted before predicting')
        
        outputs: list[list[float]] = []
        
        for Xi in X:
            if len(Xi) != self.input_nodes: 
                raise ValueError(f'Expected {self.input_nodes} inputs, got {len(Xi)}')
            
            output = Matrix.from_array(Xi)
            
            for bias, weight in zip(self.biases, self.weights):
                if bias == self.biases[-1]:
                    output = self.output_activation(weight @ output + bias)
                else:
                    output = self.activation(weight @ output + bias)
            
            if self.output_activation == ACTIVATIONS['sigmoid']:
                output = output.map(lambda x: round(x))
            elif self.output_activation == ACTIVATIONS['softmax']:
                output = output.map(lambda x: 1 if x == output.max() else 0)
               
            outputs.append(output.to_array())  
    
        return outputs
    
    def forward_pass(self, X: list[float]) -> list[Matrix]:
        ''' Forward through the neural network and return the layers activations'''
        
        if len(self.weights) == 0:
            raise ValueError('The neural network must be fitted before forward passing')
        
        if len(X) != self.input_nodes: 
            raise ValueError(f'Expected {self.input_nodes} inputs, got {len(X)}')
        
        layers: list[Matrix] = []
        layer = Matrix.from_array(X)
    
        for bias, weight in zip(self.biases, self.weights):
            if bias == self.biases[-1]:
                layer = self.output_activation(weight @ layer + bias)
            else: 
                layer = self.activation(weight @ layer + bias)
                
            layers.append(layer)

        return layers
    
    def backward_pass(self, X: list[float], y: list[float]) -> float:
        ''' Backward through the neural network updating the weights and biases and return the loss '''
        
        if len(self.weights) == 0:
            raise ValueError('The neural network must be fitted before backward passing')
        
        if len(X) != self.input_nodes: 
            raise ValueError(f'Expected {self.input_nodes} inputs, got {len(X)}')
            
        if len(y) != self.output_nodes: 
            raise ValueError(f'Expected {self.output_nodes} outputs, got {len(y)}')
        
        
        layers = self.forward_pass(X)
        
        inputs = Matrix.from_array(X)
        expected = Matrix.from_array(y)
        
        loss = (0.5 * (expected - layers[-1]) ** 2).mean()
        
        for i in range(len(layers) - 1, -1, -1):
            if i == len(layers) - 1:
                derivative = self.output_derivative(layers[i])
                delta = (expected - layers[i]) * derivative
            else:
                derivative = self.derivative(layers[i])
                delta = (self.weights[i + 1].T @ delta) * derivative
            
            bias_update = delta
            
            if i == 0:
                weight_update = delta @ inputs.T
            else:
                weight_update = delta @ layers[i - 1].T
            
            self.biases_update[i] = self.momentum * self.biases_update[i] + self.learning_rate * bias_update
            self.weights_update[i] = self.momentum * self.weights_update[i] + self.learning_rate * weight_update
    
            self.biases[i] += self.biases_update[i]
            self.weights[i] += self.weights_update[i]

        return loss

    def fit(self, X: list[list[float]], y: list[list[float]]) -> None:
        ''' Train the neural network '''
        
        if len(X) == 0 or len(y) == 0 or len(X) != len(y): 
            raise ValueError(f'Invalid data provided')
        
        self.initialize(len(X[0]), len(y[0]))
        
        best_loss: float = float('inf')
        no_change_count: int = 0
        
        for curr_iter in range(self.max_iter):
            loss_mean = 0.0
            
            for i in range(0, len(X), self.batch_size):
                batch = zip(X[i:i + self.batch_size], y[i:i + self.batch_size])
                
                for X_batch, y_batch in batch:
                    loss = self.backward_pass(X_batch, y_batch)
                    
                    loss_mean += loss
                    
            loss_mean /= len(X)
            
            if self.verbose:
                print(f'Iteration: {curr_iter + 1} | Loss: {loss_mean}')
            
            if loss_mean > best_loss - self.tol:
                no_change_count += 1
            else:
                no_change_count = 0
                
            if loss_mean < best_loss:
                best_loss = loss_mean
            
            if no_change_count >= self.max_no_change_count:
                if self.verbose:
                    print(f'No change in loss for {self.max_no_change_count} iterations, stopping')
                    
                break
            
__all__ = ['NeuralNetwork']