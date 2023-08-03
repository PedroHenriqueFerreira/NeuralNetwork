from typing import Callable
from random import uniform

class Matrix:
    ''' Matrix class '''
    
    def __init__(self, rows: int, cols: int):
        ''' Create a matrix with given rows and columns '''
        
        self.rows = rows
        self.cols = cols
        
        self.data = [[0.0 for _ in range(cols)] for _ in range(rows)]
    
    def randomize(self, bound: float = 1.0) -> 'Matrix':
        ''' Randomize the matrix values with values between -bound and bound '''

        matrix = Matrix(self.rows, self.cols)
        
        for i in range(self.rows):
            for j in range(self.cols):
                matrix.data[i][j] = uniform(-bound, bound)
                
        return matrix
    
    def zeros(self) -> 'Matrix':
        ''' Return a matrix of zeros '''
        
        matrix = Matrix(self.rows, self.cols)
        
        for i in range(self.rows):
            for j in range(self.cols):
                matrix.data[i][j] = 0.0
                
        return matrix
    
    def sum(self) -> float:
        ''' Return the sum of all elements in the matrix '''
        
        sum = 0.0
        
        for i in range(self.rows):
            for j in range(self.cols):
                sum += self.data[i][j]
        
        return sum
    
    def mean(self) -> float:
        ''' Return the mean of all elements in the matrix '''
        
        return self.sum() / (self.rows * self.cols)
    
    def map(self, func: Callable[[float], float]) -> 'Matrix':
        ''' Apply a function to each element of the matrix '''

        matrix = Matrix(self.rows, self.cols)
        
        for i in range(self.rows):
            for j in range(self.cols):
                matrix.data[i][j] = func(self.data[i][j])
    
        return matrix
    
    def __add__(self, other: 'Matrix') -> 'Matrix':
        ''' Add two matrices '''
        
        if [self.rows, self.cols] != [other.rows, other.cols]: 
            ValueError('Matrices must have compatible dimensions')
        
        matrix = Matrix(self.rows, self.cols)
        
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                matrix.data[i][j] = self.data[i][j] + other.data[i][j]
    
        return matrix
    
    def __sub__(self, other: 'Matrix') -> 'Matrix':
        ''' Subtract two matrices '''
        
        if [self.rows, self.cols] != [other.rows, other.cols]: 
            ValueError('Matrices must have compatible dimensions')
     
        matrix = Matrix(self.rows, self.cols)
        
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                matrix.data[i][j] = self.data[i][j] - other.data[i][j]
    
        return matrix
       
    def __mul__(self, other: 'Matrix') -> 'Matrix':
        ''' Hadamard product of two matrices '''
        
        if [self.rows, self.cols] != [other.rows, other.cols]:
            ValueError('Matrices must have compatible dimensions')
        
        matrix = Matrix(self.rows, self.cols)
        
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                matrix.data[i][j] = self.data[i][j] * other.data[i][j]
                
        return matrix
    
    def __rmul__(self, scale: float) -> 'Matrix':
        ''' Scalar multiplication '''
        
        matrix = Matrix(self.rows, self.cols)
            
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                matrix.data[i][j] = self.data[i][j] * scale
        
        return matrix
    
    def __matmul__(self, other: 'Matrix') -> 'Matrix':
        ''' Dot product of two matrices '''
        
        if self.cols != other.rows: 
            ValueError('Matrices must have compatible dimensions')
        
        matrix = Matrix(self.rows, other.cols)
        
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                for k in range(self.cols):
                    matrix.data[i][j] += self.data[i][k] * other.data[k][j]
        
        return matrix
    
    def __pow__(self, power: float) -> 'Matrix':
        ''' Matrix exponentiation '''
        
        matrix = Matrix(self.rows, self.cols)
        
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                matrix.data[i][j] = self.data[i][j] ** power
        
        return matrix
    
    def __str__(self) -> str:
        ''' Return a string representation for the matrix '''
        
        return str(self.data)
    
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
    
    @property
    def T(self) -> 'Matrix':
        ''' Return the transpose of the matrix '''
        
        matrix = Matrix(self.cols, self.rows)
        
        for i in range(self.rows):
            for j in range(self.cols):
                matrix.data[j][i] = self.data[i][j]
                
        return matrix
    
__all__ = ['Matrix']