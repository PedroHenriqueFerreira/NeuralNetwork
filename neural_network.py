from typing import Callable, Literal
from math import exp

from matrix import Matrix

ACTIVATIONS: dict[str, Callable[[float], float]] = {
    'identity': lambda x: x,
    'sigmoid': lambda x: 1 / (1 + exp(-x)),
    'tanh': lambda x: (exp(x) - exp(-x)) / (exp(x) + exp(-x)),
    'relu': lambda x: max(0, x),
}

DERIVATIVES: dict[str, Callable[[float], float]] = {
    'identity': lambda x: 1,
    'sigmoid': lambda x: x * (1 - x),
    'tanh': lambda x: 1 - x ** 2,
    'relu': lambda x: 1 if x > 0 else 0,
}

class NeuralNetwork:
    ''' Neural Network class '''
    
    def __init__(
        self, 
        input_nodes: int, 
        hidden_nodes: list[int], 
        output_nodes: int,
        activation: Literal['identity', 'sigmoid', 'tanh', 'relu'] = 'sigmoid',
        batch_size: int = 200, 
        learning_rate: float = 0.001,
        max_iter: int = 200
    ):
        ''' Create a neural network with the given number of input, hidden and output nodes '''
        
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        
        self.activation = activation
        
        nodes = [input_nodes] + hidden_nodes + [output_nodes]
        
        self.biases = [Matrix(n, 1).randomize() for n in nodes[1:]]
        self.weights = [Matrix(n, prev_n).randomize() for prev_n, n in zip(nodes[:-1], nodes[1:])]
    
    def forward_pass(self, X: list[float]) -> list[Matrix]:
        ''' Forward the input through the neural network '''
        
        assert len(X) == self.input_nodes, f'Expected {self.input_nodes} inputs, got {len(X)}'
        
        activations: list[Matrix] = []
        activation = Matrix.from_array(X)
        
        
        for bias, weight in zip(self.biases, self.weights):
            activation = (weight @ activation + bias).map(ACTIVATIONS[self.activation])
            activations.append(activation)

        return activations
    
    def predict(self, X: list[float]) -> list[float]:
        ''' Predict the output of the neural network '''
        
        activations = self.forward_pass(X)
        
        return activations[-1].to_array()
    
    def back_propagation(self, X: list[float], y: list[float]) -> None:
        ''' Backpropagation algorithm '''
        
        assert len(X) == self.input_nodes, f'Expected {self.input_nodes} inputs, got {len(X)}'
        assert len(y) == self.output_nodes, f'Expected {self.output_nodes} outputs, got {len(y)}'
        
        activations = self.forward_pass(X)
        
        # Backward pass
        for layer in range(len(activations) - 1, -1, -1):
            derivatives = activations[layer].map(DERIVATIVES[self.activation])
            
            if layer == len(activations) - 1:
                delta = (Matrix.from_array(y) - activations[layer]) * derivatives
            else:
                delta = (self.weights[layer + 1].T @ delta) * derivatives
            
            update_bias = delta
            
            if layer == 0:
                update_weight = delta @ Matrix.from_array(X).T
            else:
                update_weight = delta @ activations[layer - 1].T
                
            self.biases[layer] += self.learning_rate * update_bias
            self.weights[layer] += self.learning_rate * update_weight
    
    def train(self, X: list[list[float]], y: list[list[float]]) -> None:
        ''' Train the neural network '''
        
        for curr_iter in range(self.max_iter):
            err = 0.0
            
            for Xi, yi in zip(X, y):
                expected = Matrix.from_array(yi)
                output = Matrix.from_array(self.predict(Xi))
                
                err += sum([e ** 2 for e in (expected - output).to_array()])
                
            print(f'ITERATION: {curr_iter + 1} | ERROR: {err}')
            
            for i in range(0, len(X), self.batch_size):
                data = list(zip(X, y))[i:i + self.batch_size]
                
                for Xi, yi in data:
                    self.back_propagation(Xi, yi)