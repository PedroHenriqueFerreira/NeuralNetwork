from typing import Callable, Literal
from math import exp

from matrix import Matrix

def identity(matrix: Matrix) -> Matrix:
    ''' Identity activation function '''
    
    return matrix.map(lambda x: x)

def sigmoid(matrix: Matrix) -> Matrix:
    ''' Sigmoid activation function '''
    
    return matrix.map(lambda x: 1 / (1 + exp(-x)))

def tanh(matrix: Matrix) -> Matrix:
    ''' Tanh activation function '''
    
    return matrix.map(lambda x: (exp(x) - exp(-x)) / (exp(x) + exp(-x)))

def relu(matrix: Matrix) -> Matrix:
    ''' ReLU activation function '''
    
    return matrix.map(lambda x: max(0, x))

def softmax(matrix: Matrix) -> Matrix:
    ''' Softmax activation function '''
    
    sum = matrix.map(lambda x: exp(x)).sum()    
    return matrix.map(lambda x: exp(x) / sum)

ACTIVATIONS: dict[str, Callable[[Matrix], Matrix]] = {
    'identity': identity,
    'sigmoid': sigmoid,
    'tanh': tanh,
    'relu': relu,
    'softmax': softmax
}

def identity_derivative(matrix: Matrix) -> Matrix:
    ''' Identity activation function derivative '''
    
    return matrix.map(lambda x: 1)

def sigmoid_derivative(matrix: Matrix) -> Matrix:
    ''' Sigmoid activation function derivative '''
    
    return matrix.map(lambda x: x * (1 - x))

def tanh_derivative(matrix: Matrix) -> Matrix:
    ''' Tanh activation function derivative '''
    
    return matrix.map(lambda x: 1 - x ** 2)

def relu_derivative(matrix: Matrix) -> Matrix:
    ''' ReLU activation function derivative '''
    
    return matrix.map(lambda x: 1 if x > 0 else 0)

def softmax_derivative(matrix: Matrix) -> Matrix:
    ''' Softmax activation function derivative '''
    
    return matrix.map(lambda x: x * (1 - x))

DERIVATIVES: dict[str, Callable[[Matrix], Matrix]] = {
    'identity': identity_derivative,
    'sigmoid': sigmoid_derivative,
    'tanh': tanh_derivative,
    'relu': relu_derivative,
    'softmax': softmax_derivative
}

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
        tol: float = 1e-4,
        max_iter_no_change: int = 10,
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
        self.tol = tol
        self.max_iter_no_change = max_iter_no_change
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
        
        assert len(X) == self.input_nodes, f'Expected {self.input_nodes} inputs, got {len(X)}'
        
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
        
        assert len(X) == self.input_nodes, f'Expected {self.input_nodes} inputs, got {len(X)}'
        
        output = Matrix.from_array(X)
        
        for bias, weight in zip(self.biases, self.weights):
            if bias == self.biases[-1]:
                output = self.output_activation(weight @ output + bias)
            else:
                output = self.activation(weight @ output + bias)
    
        return output.to_array()
    
    def backward_pass(self, X: list[float], y: list[float]) -> float:
        ''' Backward through the neural network updating the weights and biases and return the loss '''
        
        assert len(X) == self.input_nodes, f'Expected {self.input_nodes} inputs, got {len(X)}'
        assert len(y) == self.output_nodes, f'Expected {self.output_nodes} outputs, got {len(y)}'
        
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

        return (0.5 * (expected - layers[-1]) ** 2).mean()

    def train(self, X: list[list[float]], y: list[list[float]]) -> None:
        ''' Train the neural network '''
        
        assert len(X) == len(y), f'Size of X ({len(X)}) and y ({len(y)}) does not match'
        
        best_loss: float = float('inf')
        iter_no_change_count: int = 0
        
        for curr_iter in range(self.max_iter):
            # print(f'ITERATION: {curr_iter + 1} | ERROR: {total_error / len(X)}')
            
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
                iter_no_change_count += 1
            else:
                iter_no_change_count = 0
                
            if loss_mean < best_loss:
                best_loss = loss_mean
            
            if iter_no_change_count >= self.max_iter_no_change:
                if self.verbose:
                    print(f'No improvement in the last {self.max_iter_no_change} iterations, stopping...')
                
                break