from typing import Callable

from .matrix import Matrix

def identity(matrix: Matrix) -> Matrix:
    ''' Identity activation function derivative '''
    
    return matrix.map(lambda _: 1)

def sigmoid(matrix: Matrix) -> Matrix:
    ''' Sigmoid activation function derivative '''
    
    return matrix.map(lambda y: y * (1 - y))

def tanh(matrix: Matrix) -> Matrix:
    ''' Tanh activation function derivative '''
    
    return matrix.map(lambda y: 1 - y ** 2)

def relu(matrix: Matrix) -> Matrix:
    ''' ReLU activation function derivative '''
    
    return matrix.map(lambda y: 1 if y > 0 else 0)

def softmax(matrix: Matrix) -> Matrix:
    ''' Softmax activation function derivative '''
    
    return matrix.map(lambda y: y * (1 - y))

DERIVATIVES: dict[str, Callable[[Matrix], Matrix]] = {
    'identity': identity,
    'sigmoid': sigmoid,
    'tanh': tanh,
    'relu': relu,
    'softmax': softmax
}

__all__ = ['DERIVATIVES']
