from typing import Callable
from math import exp

from .matrix import Matrix

def identity(matrix: Matrix) -> Matrix:
    ''' Identity activation function '''
    
    return matrix.map(lambda x: x)

def sigmoid(matrix: Matrix) -> Matrix:
    ''' Sigmoid activation function '''
    
    def sigmoid_x(x: float) -> float:
        ''' Sigmoid function with overflow protection '''
        
        try:
            return 1 / (1 + exp(-x))
        
        except OverflowError:
            return 0
    
    return matrix.map(sigmoid_x)

def tanh(matrix: Matrix) -> Matrix:
    ''' Tanh activation function '''
    
    def tanh_x(x: float) -> float:
        ''' Tanh function with overflow protection '''
        
        try:
            return (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        
        except OverflowError:
            return 1 if x > 0 else -1
    
    return matrix.map(tanh_x)

def relu(matrix: Matrix) -> Matrix:
    ''' ReLU activation function '''
    
    return matrix.map(lambda x: max(0, x))

def softmax(matrix: Matrix) -> Matrix:
    ''' Softmax activation function '''
    
    def softmax_x(x: float) -> float:
        ''' Softmax function with overflow protection '''
        
        try:
            exp_x = exp(x)
        except OverflowError:
            exp_x = float('inf')
        
        try:
            return exp_x / (matrix.map(lambda z: exp(z)).sum() or 1)
        except OverflowError:
            return int(exp_x == float('inf'))
    
    return matrix.map(softmax_x)

ACTIVATIONS: dict[str, Callable[[Matrix], Matrix]] = {
    'identity': identity,
    'sigmoid': sigmoid,
    'tanh': tanh,
    'relu': relu,
    'softmax': softmax
}

__all__ = ['ACTIVATIONS']