from abc import ABC, abstractmethod
from typing import Type

from .matrix import Matrix

class Optimizer(ABC):
    ''' Base class for all optimizers '''
    
    @abstractmethod
    def __init__(self, **kwargs: float | None) -> None: ...
    
    @abstractmethod
    def reset(self) -> None: ...
    
    @abstractmethod
    def update(
        self,
        biases: list[Matrix],
        weights: list[Matrix],
        biases_grads: list[Matrix],
        weights_grads: list[Matrix]
    ) -> tuple[list[Matrix], list[Matrix]]: ...
    
    @abstractmethod
    def __str__(self) -> str: ...

class SGDOptimizer(Optimizer):
    ''' Stochastic Gradient Descent optimizer '''
    
    def __init__(self, **kwargs):
        self.learning_rate = kwargs.get('learning_rate') or 0.1
        self.momentum = kwargs.get('momentum') or 0.9
        
        self.updates: list[Matrix] | None = None
    
    def reset(self):
        ''' Reset the updates '''
        
        self.updates = None
        
    def update(
        self, 
        biases: list[Matrix], 
        weights: list[Matrix], 
        biases_grads: list[Matrix], 
        weights_grads: list[Matrix]
    ) -> tuple[list[Matrix], list[Matrix]]:
        ''' Update the parameters of the neural network ''' 
        
        if self.updates is None:
            self.updates = [param.zeros() for param in biases + weights]
            
        params: list[Matrix] = biases + weights
            
        for i, grads in enumerate(biases_grads + weights_grads):
            self.updates[i] = self.momentum * self.updates[i] + (1 - self.momentum) * grads
            
            params[i] += self.learning_rate * self.updates[i]
        
        index = len(biases)
            
        return params[:index], params[index:]
    
    def __str__(self) -> str:
        ''' Return the name of the optimizer '''
        
        return 'sgd'
    
class AdamOptimizer(Optimizer):
    ''' Adaptive Moment Estimation optimizer '''
    
    def __init__(self, **kwargs):
        self.learning_rate = kwargs.get('learning_rate') or 0.001
        self.beta1 = kwargs.get('beta1') or 0.9
        self.beta2 = kwargs.get('beta2') or 0.999
        self.epsilon = kwargs.get('epsilon') or 1e-8
        
        self.ms: list[Matrix] | None = None
        self.vs: list[Matrix] | None = None
    
    def reset(self):
        ''' Reset the ms and vs '''
        
        self.ms = None
        self.vs = None
    
    def update(
        self, 
        biases: list[Matrix], 
        weights: list[Matrix], 
        biases_grads: list[Matrix], 
        weights_grads: list[Matrix]
    ) -> tuple[list[Matrix], list[Matrix]]:
        ''' Update the parameters of the neural network ''' 
        
        if len(biases) != len(biases_grads) or len(weights) != len(weights_grads):
            raise ValueError('Invalid parameters')
        
        if self.ms is None or self.vs is None:
            self.ms = [param.zeros() for param in biases + weights]
            self.vs = [param.zeros() for param in biases + weights]
        
        params: list[Matrix] = biases + weights
        
        for i, grad in enumerate(biases_grads + weights_grads):
            self.ms[i] = self.beta1 * self.ms[i] + (1 - self.beta1) * grad
            self.vs[i] = self.beta2 * self.vs[i] + (1 - self.beta2) * (grad ** 2)
            
            m_hat = (1 / (1 - self.beta1)) * self.ms[i]
            v_hat = (1 / (1 - self.beta2)) * self.vs[i]
            
            params[i] += self.learning_rate * m_hat / (self.epsilon + (v_hat ** 0.5))
            
        index = len(biases)
        
        return params[:index], params[index:]

    def __str__(self) -> str:
        ''' Return the name of the optimizer '''
        
        return 'adam'

OPTIMIZERS: dict[str, Type[Optimizer]] = {
    'sgd': SGDOptimizer,
    'adam': AdamOptimizer
}

__all__ = ['OPTIMIZERS']