from typing import Callable
from math import exp as math_exp

from matrix import Matrix

def exp(x: float) -> float:
    ''' Exponential function '''
    
    try:
        return math_exp(x)
    except OverflowError:
        # TODO: Handle overflow
        return 1e-10

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
    
    exp_sum = matrix.map(lambda x: exp(x)).sum()    
    
    if exp_sum == 0:
        exp_sum = 1
    
    return matrix.map(lambda x: exp(x) / exp_sum)

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