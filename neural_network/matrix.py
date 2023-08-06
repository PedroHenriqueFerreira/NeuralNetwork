from typing import Callable
from random import uniform

class Matrix:
    ''' Matrix class '''
    
    def __init__(self, rows: int, cols: int):
        ''' Create a matrix with given rows and columns '''
        
        self.rows = rows
        self.cols = cols
        
        self.data = [[0.0 for _ in range(cols)] for _ in range(rows)]
    
    def randomize(self, min: float = -1.0, max: float = 1.0) -> 'Matrix':
        ''' Randomize the matrix values with values between -bound and bound '''

        matrix = Matrix(self.rows, self.cols)
        
        for i in range(self.rows):
            for j in range(self.cols):
                matrix.data[i][j] = uniform(min, max)
                
        return matrix
    
    def zeros(self) -> 'Matrix':
        ''' Create a matrix with all values set to zero '''
        
        return Matrix(self.rows, self.cols)
    
    def sum(self) -> float:
        ''' Return the sum of all elements in the matrix '''
        
        sum = 0.0
        
        for i in range(self.rows):
            for j in range(self.cols):
                sum += self.data[i][j]
        
        return sum
    
    def count(self) -> int:
        ''' Return the number of elements in the matrix '''
        
        return self.rows * self.cols
    
    def mean(self) -> float:
        ''' Return the mean of all elements in the matrix '''
        
        if self.count() == 0:
            return 0.0
        
        return self.sum() / self.count()
    
    def max(self) -> float:
        ''' Return the maximum value in the matrix '''
        
        return max(self.to_array())
    
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

    def __radd__(self, value: float) -> 'Matrix':
        ''' Scalar addition '''
        
        matrix = Matrix(self.rows, self.cols)
        
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                matrix.data[i][j] = value + self.data[i][j]
                
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
   
    def __rsub__(self, value: float) -> 'Matrix':
        ''' Scalar subtraction '''
        
        matrix = Matrix(self.rows, self.cols)
        
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                matrix.data[i][j] = value - self.data[i][j]
                
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
    
    def __rmul__(self, value: float) -> 'Matrix':
        ''' Scalar multiplication '''
        
        matrix = Matrix(self.rows, self.cols)
            
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                matrix.data[i][j] = value * self.data[i][j]
        
        return matrix
    
    def __truediv__(self, other: 'Matrix') -> 'Matrix':
        ''' Divide two matrices '''
        
        if [self.rows, self.cols] != [other.rows, other.cols]:
            ValueError('Matrices must have compatible dimensions')
            
        matrix = Matrix(self.rows, self.cols)
        
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                matrix.data[i][j] = self.data[i][j] / other.data[i][j]
                
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
    
    @staticmethod
    def load(data: list[list[float]]) -> 'Matrix':
        ''' Load a matrix from a list of lists '''
        
        if len(data) == 0:
            raise ValueError('Data must not be empty')
        
        matrix = Matrix(len(data), len(data[0]))
        
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                matrix.data[i][j] = data[i][j]
        
        return matrix    
    
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