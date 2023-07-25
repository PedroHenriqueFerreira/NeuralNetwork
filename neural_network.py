from typing import Literal

from matrix import Matrix
from utils import ACTIVATIONS, DERIVATIVES

class NeuralNetwork:
    ''' Neural Network class '''
    
    def __init__(
        self, 
        input_nodes: int, 
        hidden_nodes: list[int], 
        output_nodes: int,
        activation: Literal['identity', 'sigmoid', 'tanh', 'relu'] = 'sigmoid',
        output_activation: Literal['identity', 'sigmoid', 'softmax'] = 'softmax',
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        batch_size: int = 200, 
        max_iter: int = 200,
        max_no_change_count: int = 10,
        tol: float = 1e-4,
        verbose: bool = False
    ):
        ''' Create a neural network with the given number of input, hidden and output nodes '''
        
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        
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
        
        nodes = [input_nodes] + hidden_nodes + [output_nodes]
        
        self.biases = [Matrix(n, 1).randomize() for n in nodes[1:]]
        self.weights = [Matrix(n, prev_n).randomize() for prev_n, n in zip(nodes[:-1], nodes[1:])]
        
        self.biases_update = [matrix.zeros() for matrix in self.biases]
        self.weights_update = [matrix.zeros() for matrix in self.weights]
    
    def forward_pass(self, X: list[float]) -> list[Matrix]:
        ''' Forward through the neural network and return the layers activations'''
        
        if len(X) != self.input_nodes: 
            ValueError(f'Expected {self.input_nodes} inputs, got {len(X)}')
        
        layers: list[Matrix] = []
        layer = Matrix.from_array(X)
    
        for bias, weight in zip(self.biases, self.weights):
            if bias == self.biases[-1]:
                layer = self.output_activation(weight @ layer + bias)
            else: 
                layer = self.activation(weight @ layer + bias)
                
            layers.append(layer)

        return layers
    
    def predict(self, X: list[float]) -> list[float]:
        ''' Predict the output of the neural network '''
        
        if len(X) != self.input_nodes: 
            ValueError(f'Expected {self.input_nodes} inputs, got {len(X)}')
        
        output = Matrix.from_array(X)
        
        for bias, weight in zip(self.biases, self.weights):
            if bias == self.biases[-1]:
                output = self.output_activation(weight @ output + bias)
            else:
                output = self.activation(weight @ output + bias)
    
        return output.to_array()
    
    def backward_pass(self, X: list[float], y: list[float]) -> float:
        ''' Backward through the neural network updating the weights and biases and return the loss '''
        
        if len(X) != self.input_nodes: 
            ValueError(f'Expected {self.input_nodes} inputs, got {len(X)}')
            
        if len(y) != self.output_nodes: 
            ValueError(f'Expected {self.output_nodes} outputs, got {len(y)}')
        
        layers = self.forward_pass(X)
        
        inputs = Matrix.from_array(X)
        expected = Matrix.from_array(y)
        
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

        loss = (0.5 * (expected - layers[-1]) ** 2).mean()

        return loss

    def train(self, X: list[list[float]], y: list[list[float]]) -> None:
        ''' Train the neural network '''
        
        if len(X) != len(y): 
            ValueError(f'Data and labels must have the same length')
        
        best_loss: float = float('inf')
        no_change_count: int = 0
        
        for curr_iter in range(self.max_iter):
            loss_mean = 0.0
            
            for i in range(0, len(X), self.batch_size):
                batch = zip(X[i:i + self.batch_size], y[i:i + self.batch_size])
                
                for Xi, yi in batch:
                    loss = self.backward_pass(Xi, yi)
                    
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
                    print(f'Early stopping at iteration {curr_iter + 1}')
                    
                break