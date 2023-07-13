from os import get_terminal_size
from random import random
from pprint import pformat

class Matrix:
    ''' Matrix class with some operations '''
    
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        
        self.data = [[0.0 for _ in range(cols)] for _ in range(rows)]
    
    def randomize(self):
        ''' Randomize the matrix values '''
        
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] = random() * 2 - 1    
    
    def add_bias(self):
        ''' Add bias to a single column matrix '''
        
        assert self.cols == 1, 'Matrix must have a single column'
        
        self.rows += 1
        self.data.append([1])
    
    def clone(self):
        ''' Clone the matrix '''
        
        matrix = Matrix(self.rows, self.cols)
        
        for i in range(self.rows):
            for j in range(self.cols):
                matrix.data[i][j] = self.data[i][j]

        return matrix
    
    def __add__(self, other: 'Matrix') -> 'Matrix':
        ''' Add two matrices '''
        
        assert self.rows == other.rows and self.cols == other.cols, 'Matrices must have compatible dimensions'
        
        matrix = Matrix(self.rows, self.cols)
        
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                matrix.data[i][j] = self.data[i][j] + other.data[i][j]
    
    
        return matrix
    
    def __sub__(self, other: 'Matrix') -> 'Matrix':
        ''' Subtract two matrices '''
        
        assert self.rows == other.rows and self.cols == other.cols, 'Matrices must have compatible dimensions'
        
        matrix = Matrix(self.rows, self.cols)
        
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                matrix.data[i][j] = self.data[i][j] - other.data[i][j]
    
        return matrix
    
    def __mul__(self, other: 'Matrix') -> 'Matrix':
        ''' Hadamard product of two matrices '''
        
        assert self.rows == other.rows and self.cols == other.cols, 'Matrices must have compatible dimensions'
        
        matrix = Matrix(self.rows, self.cols)
        
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                matrix.data[i][j] = self.data[i][j] * other.data[i][j]
                
        return matrix
    
    def __matmul__(self, other: 'Matrix') -> 'Matrix':
        ''' Multiply two matrices '''
        
        assert self.cols == other.rows, 'Matrices must have compatible dimensions'
        
        matrix = Matrix(self.rows, other.cols)
        
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                for k in range(self.cols):
                    matrix.data[i][j] += self.data[i][k] * other.data[k][j]
        
        return matrix
    
    def __str__(self):
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
        
        matrix = Matrix(self.cols, self.rows)
        
        for i in range(self.rows):
            for j in range(self.cols):
                matrix.data[j][i] = self.data[i][j]
                
        return matrix