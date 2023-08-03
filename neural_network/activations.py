from typing import Callable
from math import exp, inf, isnan

from neural_network.matrix import Matrix

def identity(matrix: Matrix) -> Matrix:
    ''' Identity activation function '''
    
    return matrix.map(lambda x: x)

def sigmoid(matrix: Matrix) -> Matrix:
    ''' Sigmoid activation function '''
    
    def sigmoid_(x: float) -> float:
        ''' Sigmoid function with overflow protection '''
        
        try:
            return 1 / (1 + exp(-x))
        
        except OverflowError:
            return 0
    
    return matrix.map(sigmoid_)

def tanh(matrix: Matrix) -> Matrix:
    ''' Tanh activation function '''
    
    def tanh_(x: float) -> float:
        ''' Tanh function with overflow protection '''
        
        try:
            return (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        
        except OverflowError:
            return 1 if x > 0 else -1
    
    return matrix.map(tanh_)

def relu(matrix: Matrix) -> Matrix:
    ''' ReLU activation function '''
    
    return matrix.map(lambda x: max(0, x))

def softmax(matrix: Matrix) -> Matrix:
    ''' Softmax activation function '''
    
    def exp_sum_(m: Matrix) -> float:
        ''' Exponential sum function with overflow protection '''
        
        try:
            return m.map(lambda x: exp(x)).sum()
        except OverflowError:
            return inf
    
    exp_sum = exp_sum_(matrix)
    
    def softmax_(x: float) -> float:
        ''' Softmax function with overflow protection '''
        
        try:
            return exp(x) / (exp_sum or 1)
        
        except OverflowError:
            return 1 if exp_sum == inf else inf
    
    return matrix.map(softmax_)

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

__all__ = ['ACTIVATIONS', 'DERIVATIVES']