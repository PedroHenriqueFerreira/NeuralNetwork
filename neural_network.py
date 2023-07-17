from typing import Callable
from math import exp

from matrix import Matrix

sigmoid = lambda x: 1 / (1 + exp(-x))
d_sigmoid = lambda x: x * (1 - x)

relu = lambda x: max(0, x)
d_relu = lambda x: 1 if x > 0 else 0

class NeuralNetwork:
    ''' Neural Network class '''
    
    def __init__(
        self, 
        input_nodes: int, 
        hidden_nodes: list[int], 
        output_nodes: int, 
        learning_rate: float = 0.1,
        activation: Callable[[float], float] = sigmoid,
        d_activation: Callable[[float], float] = d_sigmoid
    ):
        ''' Create a neural network with the given number of input, hidden and output nodes '''
        
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.learning_rate = learning_rate

        self.activation = activation
        self.d_activation = d_activation
        
        self.weights: list[Matrix] = []
        self.biases: list[Matrix] = []
        
        for nodes in hidden_nodes + [output_nodes]:
            prev_nodes = input_nodes if not self.weights else self.hidden_nodes[-1]
            
            weight = Matrix(nodes, prev_nodes)
            weight.randomize()
            
            bias = Matrix(nodes, 1)
            bias.randomize()
            
            self.weights.append(weight)
            self.biases.append(bias)
    
    def feed_forward(self, inputs: list[float]) -> list[float]:
        ''' Feed forward the given inputs through the neural network '''
        
        assert len(inputs) == self.input_nodes, f'Expected {self.input_nodes} inputs, got {len(inputs)}'
        
        x = Matrix.from_array(inputs)
        
        for b, w in zip(self.biases, self.weights):
            x = (w @ x + b).map(self.activation)

        return x.to_array()
    
    def back_propagate(self, x: list[float], y: list[float]) -> None:
        ''' Backpropagate the error of the neural network '''
        
        assert len(x) == self.input_nodes, f'Expected {self.input_nodes} inputs, got {len(x)}'
        assert len(y) == self.output_nodes, f'Expected {self.output_nodes} outputs, got {len(y)}'
        
        # Feed forward
        inputs = Matrix.from_array(x)
        hidden = (self.weights[0] @ inputs + self.biases[0]).map(self.activation)
        output = (self.weights[1] @ hidden + self.biases[1]).map(self.activation)
        
        # Backpropagation
        expected = Matrix.from_array(y)
        output_error = expected - output
        d_output = output.map(self.d_activation)
        
        gradient = output_error * d_output * self.learning_rate
            
        self.biases[1] += gradient
        self.weights[1] += gradient @ hidden.T
        
        hidden_error = self.weights[1].T @ output_error
        d_hidden = hidden.map(self.d_activation)
        
        gradient_hidden = hidden_error * d_hidden * self.learning_rate
        
        self.biases[0] += gradient_hidden
        self.weights[0] += gradient_hidden @ inputs.T
    
    def mutate(self, rate: float) -> None:
        ''' Mutate the weights of the neural network '''
        
        for weight in self.weights:
            weight.mutate(rate)
    
    def crossover(self, other: 'NeuralNetwork') -> 'NeuralNetwork':
        ''' Crossover two neural networks '''
        
        self_nodes = [self.input_nodes] + self.hidden_nodes + [self.output_nodes]
        other_nodes = [other.input_nodes] + other.hidden_nodes + [other.output_nodes]
        
        assert self_nodes == other_nodes, 'Neural networks must have the same number of nodes'
        
        child = NeuralNetwork(self.input_nodes, self.hidden_nodes, self.output_nodes)
        
        for i in range(len(self.weights)):
            child.weights[i] = self.weights[i].crossover(other.weights[i])
        
        return child

    def clone(self) -> 'NeuralNetwork':
        ''' Clone the neural network '''
        
        clone = NeuralNetwork(self.input_nodes, self.hidden_nodes, self.output_nodes)
        
        for i in range(len(self.weights)):
            clone.weights[i] = self.weights[i].clone()
        
        return clone
    
    @staticmethod
    def load(weights: list[Matrix]) -> 'NeuralNetwork':
        ''' Load a neural network from a list of weights '''
        
        assert len(weights) >= 2, 'Weights must not be empty'
        
        input_nodes = weights[0].cols - 1
        
        *hidden_nodes, output_nodes = [weight.rows for weight in weights]
        
        neural_network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes)
        neural_network.weights = [weight.clone() for weight in weights]
        
        return neural_network
    
nn = NeuralNetwork(2, [2], 1)

print(nn.feed_forward([1, 1]))
print(nn.feed_forward([1, 0]))
print(nn.feed_forward([0, 1]))
print(nn.feed_forward([0, 0]))

print('-----------------------')

for i in range(100000):
    nn.back_propagate([1, 0], [1])
    nn.back_propagate([0, 1], [1])
    nn.back_propagate([0, 0], [0])
    nn.back_propagate([1, 1], [0])
    
print(nn.feed_forward([1, 1]))
print(nn.feed_forward([1, 0]))
print(nn.feed_forward([0, 1]))
print(nn.feed_forward([0, 0]))