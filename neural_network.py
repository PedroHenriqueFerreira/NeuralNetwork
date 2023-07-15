from typing import Callable
from math import exp

from matrix import Matrix

sigmoid = lambda x: 1 / (1 + exp(-x))
relu = lambda x: max(0, x)

class NeuralNetwork:
    ''' Neural Network class '''
    
    def __init__(
        self, 
        input_nodes: int, 
        hidden_nodes: list[int], 
        output_nodes: int, 
        activation: Callable[[float], float] = sigmoid
    ):
        ''' Create a neural network with the given number of input, hidden and output nodes '''
        
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.activation = activation
        
        self.weights: list[Matrix] = []
        
        for nodes in hidden_nodes + [output_nodes]:
            prev_nodes = input_nodes if not self.weights else self.hidden_nodes[-1]
            
            weight = Matrix(nodes, prev_nodes + 1)
            weight.randomize()
            
            self.weights.append(weight)
    
    def feed_forward(self, inputs: list[float]) -> list[float]:
        ''' Feed forward the given inputs through the neural network '''
        
        assert len(inputs) == self.input_nodes, f'Expected {self.input_nodes} inputs, got {len(inputs)}'
        
        curr_matrix = Matrix.from_array(inputs)
        
        for weight in self.weights:
            curr_matrix.add_bias()
            curr_matrix = weight @ curr_matrix
            curr_matrix.activate(self.activation)
        
        return curr_matrix.to_array()
    
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