from typing import Callable
from math import exp

from matrix import Matrix

sigmoid = lambda x: 1 / (1 + exp(-x))
d_sigmoid = lambda x: x * (1 - x)

class NeuralNetwork:
    ''' Neural Network class '''
    
    def __init__(
        self, 
        input_nodes: int, 
        hidden_nodes: list[int], 
        output_nodes: int,
        activation: Callable[[float], float] = sigmoid,
        d_activation: Callable[[float], float] = d_sigmoid
    ):
        ''' Create a neural network with the given number of input, hidden and output nodes '''
        
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        
        self.activation = activation
        self.d_activation = d_activation
        
        self.weights: list[Matrix] = []
        
        nodes = [input_nodes] + hidden_nodes + [output_nodes]
        
        self.biases = [Matrix(n, 1).randomize() for n in nodes[1:]]
        self.weights = [Matrix(n, prev_n).randomize() for prev_n, n in zip(nodes[:-1], nodes[1:])]
    
    def feed_forward(self, inputs: list[float]) -> list[Matrix]:
        ''' Feed forward the given inputs through the neural network '''
        
        assert len(inputs) == self.input_nodes, f'Expected {self.input_nodes} inputs, got {len(inputs)}'
        
        activations: list[Matrix] = []
        
        x = Matrix.from_array(inputs)
        
        for bias, weight in zip(self.biases, self.weights):
            x = (weight @ x + bias).map(self.activation)
            activations.append(x)

        return activations
    
    def predict(self, inputs: list[float]) -> list[float]:
        ''' Predict the output of the neural network '''
        
        activations = self.feed_forward(inputs)
        
        return activations[-1].to_array()
    
    def back_propagation(self, inputs: list[float], outputs: list[float], learning_rate: float) -> None:
        ''' Backpropagation algorithm '''
        
        assert len(inputs) == self.input_nodes, f'Expected {self.input_nodes} inputs, got {len(inputs)}'
        assert len(outputs) == self.output_nodes, f'Expected {self.output_nodes} outputs, got {len(outputs)}'
        
        activations = self.feed_forward(inputs)
        
        # Backward pass
        for layer in range(len(activations) - 1, -1, -1):
            if layer == len(activations) - 1:
                delta = (Matrix.from_array(outputs) - activations[layer]) * activations[layer].map(d_sigmoid)
            else:
                delta = (self.weights[layer + 1].T @ delta) * activations[layer].map(d_sigmoid)
            
            update_bias = delta
            
            if layer == 0:
                update_weight = delta @ Matrix.from_array(inputs).T
            else:
                update_weight = delta @ activations[layer - 1].T
                
            self.biases[layer] += learning_rate * update_bias
            self.weights[layer] += learning_rate * update_weight
    
    def train(
        self, 
        data: list[tuple[list[float], list[float]]], 
        epochs: int, 
        batch_size: int = 200, 
        learning_rate: float = 0.1
    ) -> None:
        ''' Train the neural network '''
        
        for epoch in range(epochs):
            total_err = 0.0
            
            for x, y in data:
                total_err += sum([err ** 2 for err in (Matrix.from_array(self.predict(x)) - Matrix.from_array(y)).to_array()])
                
            print(f'EPOCH {epoch + 1} | ERROR: {total_err}')
            
            for i in range(0, len(data), batch_size):
                for x, y in data[i:i + batch_size]:
                    self.back_propagation(x, y, learning_rate)
    
    def mutate(self, rate: float) -> 'NeuralNetwork':
        ''' Mutate the weights of the neural network '''
        
        nn = NeuralNetwork(self.input_nodes, self.hidden_nodes, self.output_nodes)
        
        nn.biases = [bias.mutate(rate) for bias in self.biases]
        nn.weights = [weight.mutate(rate) for weight in self.weights]
            
        return nn
    
    def crossover(self, other: 'NeuralNetwork') -> 'NeuralNetwork':
        ''' Crossover two neural networks '''
        
        self_nodes = [self.input_nodes] + self.hidden_nodes + [self.output_nodes]
        other_nodes = [other.input_nodes] + other.hidden_nodes + [other.output_nodes]
        
        assert self_nodes == other_nodes, 'Neural networks must have the same number of nodes'
        
        nn = NeuralNetwork(self.input_nodes, self.hidden_nodes, self.output_nodes)
        
        nn.biases = [b.crossover(other_b) for b, other_b in zip(self.biases, other.biases)]
        nn.weights = [w.crossover(other_w) for w, other_w in zip(self.weights, other.weights)]
        
        return nn

    def clone(self) -> 'NeuralNetwork':
        ''' Clone the neural network '''
        
        nn = NeuralNetwork(self.input_nodes, self.hidden_nodes, self.output_nodes)
        
        nn.biases = [bias.clone() for bias in self.biases]
        nn.weights = [weight.clone() for weight in self.weights]
        
        return nn
    
    @staticmethod
    def load(biases: list[Matrix], weights: list[Matrix]) -> 'NeuralNetwork':
        ''' Load a neural network from a list of weights '''
        
        assert len(biases) == len(weights), 'Biases and weights must have same sizes'
        
        input_nodes = weights[0].cols
        hidden_nodes = [bias.rows for bias in biases[:-1]]
        output_nodes = biases[-1].rows
        
        nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes)
        
        nn.weights = [weight.clone() for weight in weights]
        nn.biases = [bias.clone() for bias in biases]
        
        return nn