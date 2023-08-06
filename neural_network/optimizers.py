from abc import ABC, abstractmethod

from .matrix import Matrix

class Optimizer(ABC):
    ''' Base class for all optimizers '''
    
    @abstractmethod
    def update(
        self, 
        biases: list[Matrix], 
        weights: list[Matrix], 
        biases_grads: list[Matrix], 
        weights_grads: list[Matrix]
    ) -> tuple[list[Matrix], list[Matrix]]: ...

class SGDOptimizer(Optimizer):
    ''' Stochastic Gradient Descent optimizer '''
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        self.updates: list[Matrix] | None = None
        
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
            self.updates[i] = self.momentum * self.updates[i] + self.learning_rate * grads
            
            params[i] += self.updates[i]
        
        index = len(biases)
            
        return params[:index], params[index:]
    
class AdamOptimizer(Optimizer):
    ''' Adaptive Moment Estimation optimizer '''
    
    def __init__(
        self, 
        learning_rate: float = 0.001, 
        beta1: float = 0.9, 
        beta2: float = 0.999, 
        epsilon: float = 1e-8
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.ms: list[Matrix] | None = None
        self.vs: list[Matrix] | None = None
        
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

__all__ = ['Optimizer', 'SGDOptimizer', 'AdamOptimizer']