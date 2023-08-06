from typing import Type
from abc import ABC, abstractmethod

from math import exp

from .matrix import Matrix

class Activation(ABC):
    ''' Activation function '''
    
    @abstractmethod
    def __call__(self, matrix: Matrix) -> Matrix: ...
    
    @abstractmethod
    def derivative(self, matrix: Matrix) -> Matrix: ...
    
    @abstractmethod
    def __str__(self) -> str: ...

class Identity(Activation):
    ''' Identity activation function '''
    
    def __call__(self, matrix: Matrix) -> Matrix:
        ''' Return the activation of the matrix  '''
        
        return matrix.map(lambda x: x)

    def derivative(self, matrix: Matrix) -> Matrix:
        ''' Return the matrix '''
        
        return matrix.map(lambda y: 1)
    
    def __str__(self) -> str:
        ''' Return the name of the activation '''
        
        return 'identity'

class Sigmoid(Activation):
    ''' Sigmoid activation function '''
    
    def __call__(self, matrix: Matrix) -> Matrix:
        ''' Return the activation of the matrix '''
        
        def sigmoid_x(x: float) -> float:
            ''' Sigmoid function with overflow protection '''
            
            try:
                return 1 / (1 + exp(-x))
            
            except OverflowError:
                return 0
        
        return matrix.map(sigmoid_x)
    
    def derivative(self, matrix: Matrix) -> Matrix:
        ''' Return the derivative of the matrix '''
        
        return matrix.map(lambda y: y * (1 - y))
    
    def __str__(self) -> str:
        ''' Return the name of the activation '''
        
        return 'sigmoid'

class TanH(Activation):
    ''' TanH activation function '''
    
    def __call__(self, matrix: Matrix) -> Matrix:
        ''' Return tidentityhe activation of the matrix '''
        
        def tanh_x(x: float) -> float:
            ''' TanH function with overflow protection '''
            
            try:
                return (exp(x) - exp(-x)) / (exp(x) + exp(-x))
            
            except OverflowError:
                return 1 if x > 0 else -1
        
        return matrix.map(tanh_x)
    
    def derivative(self, matrix: Matrix) -> Matrix:
        ''' Return the derivative of the matrix '''
        
        return matrix.map(lambda y: 1 - y ** 2)

    def __str__(self) -> str:
        ''' Return the name of the activation '''
        
        return 'tanh'

class ReLU(Activation):
    ''' ReLU activation function '''
    
    def __call__(self, matrix: Matrix) -> Matrix:
        ''' Return the activation of the matrix '''
        
        return matrix.map(lambda x: max(0, x))
    
    def derivative(self, matrix: Matrix) -> Matrix:
        ''' Return the derivative of the matrix '''
        
        return matrix.map(lambda y: 1 if y > 0 else 0)

    def __str__(self) -> str:
        ''' Return the name of the activation '''
        
        return 'relu'

class Softmax(Activation):
    ''' Softmax activation function '''
    
    def __call__(self, matrix: Matrix) -> Matrix:
        ''' Return the activation of the matrix '''
        
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
    
    def derivative(self, matrix: Matrix) -> Matrix:
        ''' Return the derivative of the matrix '''
        
        return matrix.map(lambda y: y * (1 - y))

    def __str__(self) -> str:
        ''' Return the name of the activation '''
        
        return 'softmax'

ACTIVATIONS: dict[str, Type[Activation]] = {
    'identity': Identity,
    'sigmoid': Sigmoid,
    'tanh': TanH,
    'relu': ReLU,
    'softmax': Softmax
}

__all__ = ['ACTIVATIONS']