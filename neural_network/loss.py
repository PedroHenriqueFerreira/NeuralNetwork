from abc import ABC, abstractmethod

from typing import Type
from math import log

from .matrix import Matrix

class Loss(ABC):
    ''' Loss function '''
    
    @abstractmethod
    def __call__(self, y_true: Matrix, y_pred: Matrix) -> float: ...
    
    @abstractmethod
    def derivative(self, y_true: Matrix, y_pred: Matrix) -> Matrix: ...
    
    @abstractmethod
    def __str__(self) -> str: ...    

class SquaredLoss(Loss):
    ''' Squared loss function '''
    
    def __call__(self, y_true: Matrix, y_pred: Matrix) -> float:
        ''' Return the loss of the matrix '''
        
        return (0.5 * (y_true - y_pred) ** 2).mean()

    def derivative(self, y_true: Matrix, y_pred: Matrix) -> Matrix:
        ''' Return the derivative of the matrix '''
        
        return y_true - y_pred
    
    def __str__(self) -> str:
        ''' Return the name of the loss '''
        
        return 'mse'

class LogLoss(Loss):
    ''' Cross entropy loss function '''

    def __call__(self, y_true: Matrix, y_pred: Matrix) -> float:
        ''' Return the loss of the matrix '''
        
        y_pred = y_pred.map(lambda x: max(1e-15, min(1 - 1e-15, x)))
        
        y_pred_log = y_pred.map(lambda x: log(x))
        y_pred_neg_log = y_pred.map(lambda x: log(1 - x))
        
        return - (y_true * y_pred_log + (1 - y_true) * y_pred_neg_log).mean()

    def derivative(self, y_true: Matrix, y_pred: Matrix) -> Matrix:
        ''' Return the derivative of the matrix '''
        
        y_pred = y_pred.map(lambda x: max(1e-15, min(1 - 1e-15, x)))
        
        return y_true / y_pred - (1 - y_true) / (1 - y_pred)

    def __str__(self) -> str:
        ''' Return the name of the loss '''
        
        return 'log'

LOSS: dict[str, Type[Loss]] = {
    'mse': SquaredLoss,
    'log': LogLoss
}

__all__ = ['LOSS']