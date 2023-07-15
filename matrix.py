from typing import Callable, Union
from random import random, randint

from os import get_terminal_size
from pprint import pformat

class Matrix:
    ''' Matrix class with some operations '''
    
    def __init__(self, rows: int, cols: int):
        ''' Create a matrix with given rows and columns '''
        
        self.rows = rows
        self.cols = cols
        
        self.data = [[0.0 for _ in range(cols)] for _ in range(rows)]
    
    def randomize(self) -> None:
        ''' Randomize the matrix values '''
        
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] = random() * 2 - 1    
    
    def mutate(self, rate: float) -> None:
        ''' Mutate the matrix values with a given rate '''
        
        for i in range(self.rows):
            for j in range(self.cols):
                if random() < rate:
                    new_val = self.data[i][j] + random() * 2 - 1
                    self.data[i][j] = max(-1, min(1, new_val))

    def crossover(self, other: 'Matrix') -> 'Matrix':
        ''' Crossover two matrices '''
        
        child = Matrix(self.rows, self.cols)
        
        rand_row = randint(0, self.rows - 1)
        rand_col = randint(0, self.cols - 1)
        
        for i in range(self.rows):
            for j in range(self.cols):
                if i < rand_row or (i == rand_row and j <= rand_col):
                    child.data[i][j] = self.data[i][j]
                else:
                    child.data[i][j] = other.data[i][j]
        
        return child
    
    def activate(self, func: Callable[[float], float]) -> None:
        
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] = func(self.data[i][j])
    
    def add_bias(self) -> None:
        ''' Add bias to a single column matrix '''
        
        assert self.cols == 1, 'Matrix must have a single column'
        
        self.rows += 1
        self.data.append([1])
    
    def clone(self) -> 'Matrix':
        ''' Clone the matrix '''
        
        clone = Matrix(self.rows, self.cols)
        
        for i in range(self.rows):
            for j in range(self.cols):
                clone.data[i][j] = self.data[i][j]

        return clone
    
    def __add__(self, other: 'Matrix') -> 'Matrix':
        ''' Add two matrices '''
        
        assert [self.rows, self.cols] == [other.rows, other.cols], 'Matrices must have compatible dimensions'
        
        result = Matrix(self.rows, self.cols)
        
        for i in range(result.rows):
            for j in range(result.cols):
                result.data[i][j] = self.data[i][j] + other.data[i][j]
    
    
        return result
    
    def __sub__(self, other: 'Matrix') -> 'Matrix':
        ''' Subtract two matrices '''
        
        assert [self.rows, self.cols] == [other.rows, other.cols], 'Matrices must have compatible dimensions'
        
        result = Matrix(self.rows, self.cols)
        
        for i in range(result.rows):
            for j in range(result.cols):
                result.data[i][j] = self.data[i][j] - other.data[i][j]
    
        return result
       
    def __mul__(self, other: Union['Matrix', int, float]) -> 'Matrix':
        ''' Hadamard product of two matrices or scalar multiplication '''
        
        if isinstance(other, int | float):
            result = Matrix(self.rows, self.cols)
            
            for i in range(result.rows):
                for j in range(result.cols):
                    result.data[i][j] = self.data[i][j] * other
            
            return result
        
        assert [self.rows, self.cols] == [other.rows, other.cols], 'Matrices must have compatible dimensions'
        
        result = Matrix(self.rows, self.cols)
        
        for i in range(result.rows):
            for j in range(result.cols):
                result.data[i][j] = self.data[i][j] * other.data[i][j]
                
        return result
    
    def __matmul__(self, other: 'Matrix') -> 'Matrix':
        ''' Multiply two matrices '''
        
        assert self.cols == other.rows, 'Matrices must have compatible dimensions'
        
        result = Matrix(self.rows, other.cols)
        
        for i in range(result.rows):
            for j in range(result.cols):
                for k in range(self.cols):
                    result.data[i][j] += self.data[i][k] * other.data[k][j]
        
        return result
    
    def __str__(self) -> str:
        ''' Return a string representation for the matrix '''
        
        return pformat(self.data, width=get_terminal_size().columns)
    
    def to_array(self) -> list[float]:
        ''' Return a list representation for the matrix '''
        
        array = []
        
        for i in range(self.rows):
            for j in range(self.cols):
                array.append(self.data[i][j])
        
        return array
    
    @staticmethod
    def from_array(arr: list[float]) -> 'Matrix':
        ''' Create a single column matrix from a list '''
        
        matrix = Matrix(len(arr), 1)
        
        for i in range(len(arr)):
            matrix.data[i][0] = arr[i]
        
        return matrix    
        
    @staticmethod
    def load(data: list[list[float]]) -> 'Matrix':
        ''' Load a matrix from a list of lists '''
        
        rows = len(data)
        cols = len(data[0])
        
        matrix = Matrix(rows, cols)
        
        for i in range(rows):
            for j in range(cols):
                matrix.data[i][j] = data[i][j]
        
        return matrix
    
    @property
    def T(self) -> 'Matrix':
        ''' Return the transpose of the matrix '''
        
        transpose = Matrix(self.cols, self.rows)
        
        for i in range(self.rows):
            for j in range(self.cols):
                transpose.data[j][i] = self.data[i][j]
                
        return transpose