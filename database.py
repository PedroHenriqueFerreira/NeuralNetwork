import pandas as pd

from typing import Any, Callable, Union

class Database:
    ''' Data base class '''
    
    def __init__(self, columns: list[Any], values: list[list[Any]]):
        ''' Create a new data base instance with the given columns and values '''
        
        if len(set(columns)) != len(columns):
            raise ValueError('Columns must be unique')
        
        if len(set([len(row) for row in values])) > 1:
            raise ValueError('All rows must have the same size')  
        
        if len(values) > 0 and len(columns) != len(values[0]):
            raise ValueError('Number of columns must be equal to the number of rows')
        
        self.columns = columns
        self.values = values
    
    def __str__(self):
        ''' Return a string representation of the data base '''
        
        columns_width: list[int] = []
        
        for i in range(len(self.columns)):
            width = max([len(str(row[i])) for row in [self.columns, *self.values]])
            
            columns_width.append(width)
        
        lines: list[str] = []
        
        for row in [self.columns, *self.values]:
            lines.append('  '.join([str(item).rjust(columns_width[i]) for i, item in enumerate(row)]))
            
        return '\n'.join(lines)

    def map(self, func: Callable[[Any], Any]) -> 'Database':
        ''' Map the data base with the given function '''
        
        values = [[func(item) for item in row] for row in self.values]
        
        return Database(self.columns.copy(), values)

    def __gt__(self, other: int | float) -> 'Database':
        ''' Return if the data base is greater than the given value '''
        
        return self.map(lambda x: x > other)

    def __lt__(self, other: int | float) -> 'Database':
        ''' Return if the data base is less than the given value '''
        
        return self.map(lambda x: x < other)
    
    def __ge__(self, other: int | float) -> 'Database':
        ''' Return if the data base is greater than or equal to the given value '''
        
        return self.map(lambda x: x >= other)
    
    def __le__(self, other: int | float) -> 'Database':
        ''' Return if the data base is less than or equal to the given value '''
        
        return self.map(lambda x: x <= other)
    
    def __eq__(self, other: int | float) -> 'Database': # type: ignore
        ''' Return if the data base is equal to the given value '''
        
        return self.map(lambda x: x == other)
    
    def __ne__(self, other: int | float) -> 'Database':  # type: ignore
        ''' Return if the data base is not equal to the given value '''
        
        return self.map(lambda x: x != other)

    def is_in(self, other: list[Any]) -> 'Database':
        ''' Return if the data base is in the given list '''
        
        return self.map(lambda x: x in other)

    def __and__(self, other: 'Database') -> 'Database':
        ''' Return the & operator between the data base and the given data base '''

        if len(self.values) != len(other.values):
            raise ValueError('Data bases must have the same size') 
        
        values: list[Any] = []
        
        for row, other_row in zip(self.values, other.values):
            is_self_true = all([item is True for item in row])
            is_other_true = all([item is True for item in other_row])
            
            if is_self_true and is_other_true:
                values.append([True])
            else:
                values.append([False])

        return Database([None], values)
    
    def __or__(self, other: 'Database') -> 'Database':
        ''' Return the | operator between the data base and the given data base '''

        if len(self.values) != len(other.values):
            raise ValueError('Data bases must have the same size') 
        
        values: list[Any] = []
        
        for row, other_row in zip(self.values, other.values):
            is_self_true = all([item is True for item in row])
            is_other_true = all([item is True for item in other_row])
            
            if is_self_true or is_other_true:
                values.append([True])
            else:
                values.append([False])

        return Database([None], values)

    def __getitem__(self, key: Union[str, int, slice, list[Any], tuple[Any, ...], 'Database']) -> 'Database':
        ''' Filter the data base with the given key '''
        
        if isinstance(key, str):
            if key not in self.columns:
                raise ValueError(f'Column {key} not found')
            
            index = self.columns.index(key)
            
            return Database([key], [[row[index]] for row in self.values])
        
        elif isinstance(key, int):
            if key >= len(self.columns):
                raise ValueError(f'Column index {key} out of range')
    
            return Database([self.columns[key]], [[row[key]] for row in self.values])
        
        elif isinstance(key, list | tuple):
            indexes: dict[Any, int] = {}
        
            for column in key:
                if isinstance(column, int):
                    if column >= len(self.columns):
                        raise ValueError(f'Column index {column} out of range')
                    
                    indexes[self.columns[column]] = column
                    
                elif isinstance(column, str):
                    if column not in self.columns:
                        raise ValueError(f'Column {column} not found')
                    
                    indexes[column] = self.columns.index(column)
                    
                else:
                    raise ValueError(f'Invalid column {column}')
            
            values: list[list[Any]] = []
            
            for row in self.values:            
                values.append([row[index] for index in indexes.values()])
            
            columns = list(indexes.keys())
            
            return Database(columns, values)

        elif isinstance(key, slice):
            return Database(self.columns[key], [row[key] for row in self.values])
        
        elif isinstance(key, Database):
            if len(self.values) != len(key.values):
                raise ValueError('Data bases must have the same size')
            
            values = []
            
            for row, other_row in zip(self.values, key.values):
                if any([item is not False and item is not True for item in other_row]):
                    raise ValueError('Data base must be a boolean data base')
                
                if any([item is False for item in other_row]):
                    continue
                
                values.append(row.copy())
            
            return Database(self.columns.copy(), values)
        
        else:
            raise ValueError(f'Invalid key {key}')
    
    def __setitem__(self, key: str | int, value: Union[list[Any] | tuple[Any, ...], 'Database']) -> None:
        ''' Set the given column with the given value '''
        
        if isinstance(key, str):
            if key not in self.columns:
                raise ValueError(f'Column {key} not found')
            
            index = self.columns.index(key)
        
        elif isinstance(key, int):
            if key >= len(self.columns):
                raise ValueError(f'Column index {key} out of range')
            
            index = key
        
        if isinstance(value, Database):
            if len(self.values) != len(value.values):
                raise ValueError('Data bases must have the same rows size')
            
            for row, other_row in zip(self.values, value.values):
                row[index] = other_row[0]
            
        elif isinstance(value, list | tuple):
            if len(value) != len(self.values):
                raise ValueError('Value must have the same size as the data base rows')
            
            for row, item in zip(self.values, value):
                row[index] = item

    @staticmethod
    def read_csv(path: str, separator: str = ',', columns: list[Any] | None = []) -> 'Database':
        ''' Read a csv file and return a data base instance '''

        values: list[list[Any]] = []
        
        with open(path, 'r') as f:
            for line in f.readlines():
                items = line.strip().split(separator)

                row: list[Any] = []                

                for item in items:
                    item = item.strip()
                    
                    if item == '':
                        row.append(None)
                    elif item.isnumeric():
                        row.append(int(item))
                    elif item.replace('.', '', 1).isnumeric():
                        row.append(float(item))
                    else:
                        row.append(item)
                
                if columns is None:
                    columns = list(range(len(row)))
                    
                if len(columns) == 0:
                    columns.extend(row)
                else:
                    values.append(row)
        
        if columns is None:
            columns = []
            
        return Database(columns, values)
    
    def sum(self) -> float:
        ''' Return the sum of all items in the data base '''
        
        return sum([sum(row) for row in self.values]) # type: ignore
    
    def count(self) -> int:
        ''' Return the number of items in the data base '''
        
        return len(self.values) * len(self.columns)
    
    def mean(self) -> float:
        ''' Return the mean of all values in the data base '''
        
        if self.count() == 0:
            return 0
        
        return self.sum() / self.count()
    
    def std(self) -> float:
        ''' Return the standard deviation of all values in the data base '''
        
        if self.count() == 0:
            return 0
        
        mean = self.mean()   
             
        var = sum([(item - mean) ** 2 for row in self.values for item in row]) / self.count()
        
        std: float = var ** 0.5
        
        return std if std != 0 else 1