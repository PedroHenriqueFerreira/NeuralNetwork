import json

from typing import Literal

from .matrix import *

from .optimizers import *
from .activations import *
from .loss import *

class NeuralNetwork:
    ''' Neural Network class '''
    
    def __init__(
        self,  
        hidden_nodes: list[int] = [100],
        activation: Literal['identity', 'sigmoid', 'tanh', 'relu'] = 'relu',
        output_activation: Literal['identity', 'sigmoid', 'softmax'] = 'softmax',
        optimizer: Literal['sgd', 'adam'] = 'adam',
        loss: Literal['mse', 'log'] = 'log',
        batch_size: int = 200, 
        max_iter: int = 200,
        max_no_change_count: int = 10,
        tol: float = 1e-4,
        verbose: bool = False,
        
        # Optimizer parameters
        learning_rate: float | None = None, # 0.1 for sgd, 0.001 for adam
        momentum: float | None = None, # 0.9
        beta1: float | None = None, # 0.9
        beta2: float | None = None, # 0.999
        epsilon: float | None = None # 1e-8
    ):
        
        ''' Create a neural network with the given number of input, hidden and output nodes '''
        
        self.input_nodes: int = 0
        self.output_nodes: int = 0
        
        self.hidden_nodes = hidden_nodes
        
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.max_no_change_count = max_no_change_count
        self.tol = tol
        self.verbose = verbose
        
        self.biases: list[Matrix] = []
        self.weights: list[Matrix] = []
        
        self.activation = ACTIVATIONS[activation]()
        self.output_activation = ACTIVATIONS[output_activation]()
        
        self.optimizer_params = {
            'learning_rate': learning_rate,
            'momentum': momentum,
            'beta1': beta1,
            'beta2': beta2,
            'epsilon': epsilon
        }
        
        self.optimizer = OPTIMIZERS[optimizer](**self.optimizer_params)
        
        self.loss = LOSS[loss]()
                
    def initialize(self, input_nodes: int, output_nodes: int) -> None:
        ''' Initialize the weights and biases of the neural network '''
        
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        
        nodes = [input_nodes] + self.hidden_nodes + [output_nodes]
        
        self.biases.clear()
        self.weights.clear()
        
        for n_in, n_out in zip(nodes[:-1], nodes[1:]):
            bound = (1 / n_in) ** 0.5
            
            self.biases.append(Matrix(n_out, 1).randomize(-bound, bound))
            self.weights.append(Matrix(n_out, n_in).randomize(-bound, bound))
    
    def predict(self, X: list[list[float]]) -> list[list[float]]:
        ''' Predict the output of the neural network '''
        
        if len(self.weights) == 0:
            raise ValueError('The neural network must be fitted before predicting')
        
        predictions: list[list[float]] = []
        
        for Xi in X:
            if len(Xi) != self.input_nodes: 
                raise ValueError(f'Expected {self.input_nodes} inputs, got {len(Xi)}')
            
            output = Matrix.from_array(Xi)
            
            for bias, weight in zip(self.biases, self.weights):
                if bias == self.biases[-1]:
                    output = self.output_activation(weight @ output + bias)
                    
                    match str(self.output_activation):
                        case 'sigmoid':
                            output = output.map(lambda x: round(x))
                        case 'softmax':
                            output = output.map(lambda x: 1 if x == output.max() else 0)
                        
                else:
                    output = self.activation(weight @ output + bias)
               
            predictions.append(output.to_array())  
    
        return predictions
    
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
        
        biases_grads: list[Matrix] = [bias.zeros() for bias in self.biases]
        weights_grads: list[Matrix] = [weight.zeros() for weight in self.weights]
        
        loss = self.loss(expected, layers[-1])
        
        for i in range(len(layers) - 1, -1, -1):
            if i == len(layers) - 1:
                output_activation_derivative = self.output_activation.derivative(layers[i])
                loss_derivative = self.loss.derivative(expected, layers[i])
                
                delta = loss_derivative * output_activation_derivative
            else:
                activation_derivative = self.activation.derivative(layers[i])
                
                delta = (self.weights[i + 1].T @ delta) * activation_derivative
            
            biases_grads[i] = delta
            
            if i == 0:
                weights_grads[i] = delta @ inputs.T
            else: 
                weights_grads[i] = delta @ layers[i - 1].T

        self.biases, self.weights = self.optimizer.update(self.biases, self.weights, biases_grads, weights_grads)

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
    
    def accuracy(self, X: list[list[float]], y: list[list[float]]) -> float:
        ''' Return the accuracy of the neural network '''
        
        if len(X) == 0 or len(y) == 0 or len(X) != len(y): 
            raise ValueError(f'Invalid data provided')
        
        predictions = self.predict(X)
        
        return sum(yi == y_pred for yi, y_pred in zip(y, predictions)) / len(y)
    
    def to_json(self, path: str) -> None:
        ''' Save the neural network to a json file '''
        
        attributes = self.__dict__.copy()
        
        for key in attributes:
            if key in ['biases', 'weights']:
                attributes[key] = [matrix.data for matrix in attributes[key]]

            if key in ['activation', 'output_activation', 'loss', 'optimizer']:
                attributes[key] = str(attributes[key])
                
        json.dump(attributes, open(path, 'w'))
            
    @staticmethod
    def from_json(path: str) -> 'NeuralNetwork':
        ''' Load the neural network from a json file '''
        
        attributes = json.load(open(path, 'r'))
        
        for key in attributes:
            match key:
                case 'biases' | 'weights':
                    attributes[key] = [Matrix.load(data) for data in attributes[key]]
                    
                case 'activation' | 'output_activation':
                    attributes[key] = ACTIVATIONS[attributes[key]]()
        
                case 'optimizer':
                    attributes[key] = OPTIMIZERS[attributes[key]](**attributes['optimizer_params'])
                    
                case 'loss':
                    attributes[key] = LOSS[attributes[key]]() 
                
        nn = NeuralNetwork()
        nn.__dict__.update(attributes)
        
        return nn
   
     
__all__ = ['NeuralNetwork']