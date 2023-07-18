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
        output_nodes: int
    ):
        ''' Create a neural network with the given number of input, hidden and output nodes '''
        
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        
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
        
        a = Matrix.from_array(inputs)
        
        for b, w in zip(self.biases, self.weights):
            a = (w @ a + b).map(sigmoid)

        return a.to_array()
    
    def back_propagation(self, inputs: list[float], outputs: list[float]) -> tuple[list[Matrix], list[Matrix]]:
        ''' Backpropagation algorithm '''
        
        assert len(inputs) == self.input_nodes, f'Expected {self.input_nodes} inputs, got {len(inputs)}'
        assert len(outputs) == self.output_nodes, f'Expected {self.output_nodes} outputs, got {len(outputs)}'
        
        x = Matrix.from_array(inputs)
        y = Matrix.from_array(outputs)
        
        # Store activations
        A = []
        
        # Forward pass
        a = x
        
        for b, w in zip(self.biases, self.weights):
            a = (w @ a + b).map(sigmoid)
            A.append(a)
        
        dB = [b.map(lambda _: 0) for b in self.biases]
        dW = [w.map(lambda _: 0) for w in self.weights]
        
        # Backward pass
        for l in range(len(A) - 1, -1, -1):
            if l == len(A) - 1:
                delta = (A[l] - y) * A[l].map(d_sigmoid)
            else:
                delta = (self.weights[l + 1].T @ delta) * A[l].map(d_sigmoid)
            
            if l == 0:
                dW[l] = delta @ x.T
            else:
                dW[l] = delta @ A[l - 1].T
                
            dB[l] = delta
                
        return dW, dB    
    
    def gradient_descent(self, batch: list[tuple[list[float], list[float]]], eta: float) -> None:
        ''' Stochastic gradient descent '''
         
        dB_total = [b.map(lambda _: 0) for b in self.biases]
        dW_total = [w.map(lambda _: 0) for w in self.weights]
        
        for x, y in batch:
            dW, dB = self.back_propagation(x, y)
            
            dB_total = [b + db for b, db in zip(dB_total, dB)]
            dW_total = [w + dw for w, dw in zip(dW_total, dW)]
    
        rate = eta / len(batch)
    
        self.biases = [b - (rate * db_total) for b, db_total in zip(self.biases, dB_total)]
        self.weights = [w - (rate * dw_total) for w, dw_total in zip(self.weights, dW_total)]
    
    def train(
        self, 
        data: list[tuple[list[float], list[float]]], 
        epochs: int, 
        batch_size: int = 64, 
        eta: float = 3
    ) -> None:
        ''' Train the neural network '''
        
        for _ in range(epochs):
            for i in range(0, len(data), batch_size):
                self.gradient_descent(data[i:i + batch_size], eta)
    
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
    
nn = NeuralNetwork(2, [2, 2], 1)

nn.train([
    ([1, 1], [0]),
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
], 10000)

print(nn.feed_forward([1, 1]))
print(nn.feed_forward([0, 0]))
print(nn.feed_forward([0, 1]))
print(nn.feed_forward([1, 0]))