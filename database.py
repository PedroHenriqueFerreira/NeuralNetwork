import pandas as pd

from typing import Any, Callable, Union

class Database:
    ''' Data base class '''
    
    def __init__(self, columns: list[str], values: list[list[Any]]):
        ''' Create a new data base instance with the given columns and values '''
        
        if len(set(columns)) != len(columns):
            raise ValueError('Columns must be unique')
        
        if len(values) > 0 and len(columns) != len(values[0]):
            raise ValueError('Columns and values are not compatible')
        
        if len(set([len(row) for row in values])) > 1:
            raise ValueError('Invalid values shape')
        
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
        
        return Database(self.columns[:], values)

    def __gt__(self, other: int | float) -> 'Database':
        ''' Return if the data base values is greater than the given value '''
        
        return self.map(lambda x: x > other)

    def __lt__(self, other: int | float) -> 'Database':
        ''' Return if the data base values is less than the given value '''
        
        return self.map(lambda x: x < other)
    
    def __ge__(self, other: int | float) -> 'Database':
        ''' Return if the data base values is greater than or equal to the given value '''
        
        return self.map(lambda x: x >= other)
    
    def __le__(self, other: int | float) -> 'Database':
        ''' Return if the data base values is less than or equal to the given value '''
        
        return self.map(lambda x: x <= other)
    
    def __eq__(self, other: int | float) -> 'Database': # type: ignore
        ''' Return if the data base values is equal to the given value '''
        
        return self.map(lambda x: x == other)
    
    def __ne__(self, other: int | float) -> 'Database':  # type: ignore
        ''' Return if the data base values is not equal to the given value '''
        
        return self.map(lambda x: x != other)

    def is_in(self, other: list[Any]) -> 'Database':
        ''' Return if the data base values is in the given list '''
        
        return self.map(lambda x: x in other)

    def __and__(self, other: 'Database') -> 'Database':
        ''' Return the & operator between the data base and the given data base '''

        if len(self.values) != len(other.values):
            raise ValueError('Invalid data base shape') 
        
        values = [[all(row) & all(other_row)] for row, other_row in zip(self.values, other.values)]
        
        return Database(['and'], values)
    
    def __or__(self, other: 'Database') -> 'Database':
        ''' Return the | operator between the data base and the given data base '''

        if len(self.values) != len(other.values):
            raise ValueError('Invalid data base shape') 
        
        values = [[all(row) | all(other_row)] for row, other_row in zip(self.values, other.values)]
        
        return Database(['or'], values)

    def __xor__(self, other: 'Database') -> 'Database':
        ''' Return the ^ operator between the data base and the given data base '''

        if len(self.values) != len(other.values):
            raise ValueError('Invalid data base shape') 
        
        values = [[all(row) ^ all(other_row)] for row, other_row in zip(self.values, other.values)]
        
        return Database(['xor'], values)

    def __getitem__(
        self, 
        key: Union[str, int, slice, list[str | int], tuple[str | int, ...], 'Database']
    ) -> 'Database':
        ''' Filter the data base with the given key '''
        
        if isinstance(key, str):
            if key not in self.columns:
                raise ValueError(f'Column {key} not found')
            
            index = self.columns.index(key)
            
            return Database([key], [[row[index]] for row in self.values])
        
        elif isinstance(key, int):
            if key < 0 or key >= len(self.columns):
                raise ValueError(f'Column index {key} out of range')
    
            return Database([self.columns[key]], [[row[key]] for row in self.values])
        
        elif isinstance(key, list | tuple):
            indexes: dict[str, int] = {}
        
            for column in key:
                if isinstance(column, int):
                    if column < 0 or column >= len(self.columns):
                        raise ValueError(f'Column index {column} out of range')
                    
                    indexes[self.columns[column]] = column
                    
                elif isinstance(column, str):
                    if column not in self.columns:
                        raise ValueError(f'Column {column} not found')
                    
                    indexes[column] = self.columns.index(column)
                    
                else:
                    raise ValueError(f'Invalid column {column}')
            
            return Database(list(indexes.keys()), [[row[i] for i in indexes.values()] for row in self.values])

        elif isinstance(key, slice):
            values = [key.start, key.stop, key.step]
            
            if isinstance(values[-1], str):
                raise ValueError(f'Invalid step {values[-1]}')
            
            for i, value in enumerate(values):
                if isinstance(value, str):
                    if value not in self.columns:
                        raise ValueError(f'Column {value} not found')
                    
                    values[i] = self.columns.index(value)
                elif not isinstance(value, int) and value is not None:
                    raise ValueError(f'Invalid index {value}')
            
            return Database(self.columns[slice(*values)], [row[slice(*values)] for row in self.values])
        
        elif isinstance(key, Database):
            if len(self.values) != len(key.values):
                raise ValueError('Invalid data base shape')
            
            for column in key.columns:
                if column in self.columns + ['and', 'or', 'xor']:
                    continue
                
                raise ValueError(f'Invalid data base columns')
            
            columns = self.columns[:]
            rows = [row[:] for row, other_row in zip(self.values, key.values) if all(other_row)]
             
            return Database(columns, rows)
        
        else:
            raise ValueError(f'Invalid key {key}')
    
    def __setitem__(
        self, 
        key: str | int | slice, value: Union[list[Any], tuple[Any, ...], 'Database']
    ) -> None:
        ''' Set the given column with the given value '''
        
        index = 0
        column: str = ''
        
        if isinstance(key, str):
            if key not in self.columns:
                index = len(self.columns)
            else:
                index = self.columns.index(key)
                
            column = key
                
        elif isinstance(key, int):
            if key < 0 or key >= len(self.columns):
                raise ValueError(f'Column index {key} out of range')
            
            index = key
            
            column = self.columns[key]
            
        elif isinstance(key, slice):
            if isinstance(key.start, str):
                if key.start not in self.columns:
                    raise ValueError(f'Column {key.start} not found')
                
                index = self.columns.index(key.start)
            elif isinstance(key.start, int):
                if key.start < 0 or key.start >= len(self.columns):
                    raise ValueError(f'Column index {key.start} out of range')
                
                index = key.start
            elif key.start is not None:
                raise ValueError(f'Invalid slice start {key.start}')
            
            column = self.columns[index]
        else:
            raise ValueError(f'Invalid key {key}')
            
        if index < len(self.columns):
            del self[key]
            
        if isinstance(value, Database):
            if len(self.values) != len(value.values):
                raise ValueError('Invalid data base shape')
            
            column_index = index
            
            for column in value.columns:
                if column in self.columns:
                    raise ValueError(f'Column {column} already exists')
                
                self.columns.insert(column_index, column)
                
                column_index += 1
            
            row_index = index
                
            for row, other_row in zip(self.values, value.values):
                for other_item in other_row:
                    row.insert(row_index, other_item)
                    
                    row_index += 1
                    
                row_index = index
            
        elif isinstance(value, list | tuple):
            if len(value) != len(self.values):
                raise ValueError('Invalid data shape')
            
            self.columns.insert(index, column)
            
            for row, item in zip(self.values, value):
                row.insert(index, item)
             
        else:
            raise ValueError(f'Invalid value {value}')

    def __delitem__(self, key: str | int | slice | list[str | int] | tuple[str | int, ...]) -> None:
        ''' Delete the given column '''
        
        if isinstance(key, str):
            if key not in self.columns:
                raise ValueError(f'Column {key} not found')
            
            index = self.columns.index(key)
            
            del self.columns[index]
            
            for row in self.values:
                del row[index]
        
        elif isinstance(key, int):
            if key >= len(self.columns):
                raise ValueError(f'Column index {key} out of range')
        
            del self.columns[key]
            
            for row in self.values:
                del row[key]

        elif isinstance(key, list | tuple):
            columns: list[str] = []
        
            for column in key:
                if isinstance(column, int):
                    if column < 0 or column >= len(self.columns):
                        raise ValueError(f'Column index {column} out of range')
                    
                    columns.append(self.columns[column])
                    
                elif isinstance(column, str):
                    if column not in self.columns:
                        raise ValueError(f'Column {column} not found')
                    
                    columns.append(column)
                    
                else:
                    raise ValueError(f'Invalid column {column}')
            
            for column in columns:
                index = self.columns.index(column)
                
                del self.columns[index]
                
                for row in self.values:
                    del row[index]
                    
        elif isinstance(key, slice):
            values = [key.start, key.stop, key.step]
            
            if isinstance(values[-1], str):
                raise ValueError(f'Invalid step {values[-1]}')
            
            for i, value in enumerate(values):
                if isinstance(value, str):
                    if value not in self.columns:
                        raise ValueError(f'Column {value} not found')
                    
                    values[i] = self.columns.index(value)
                elif not isinstance(value, int) and value is not None:
                    raise ValueError(f'Invalid index {value}')
            
            del self.columns[slice(*values)]
            
            for row in self.values:
                del row[slice(*values)]
        
        else:
            raise ValueError(f'Invalid key {key}')
        
    def unique(self) -> list[Any]:
        ''' Return the unique values of the data base '''
        
        values: list[Any] = []
        
        for row in self.values:
            for item in row:
                if item in values:
                    continue
                
                values.append(item)
        
        return values
    
    def sum(self) -> float:
        ''' Return the sum of all items in the data base '''
        
        result = 0.0
        
        for row in self.values:
            for item in row:
                if not isinstance(item, int | float | bool):
                    raise ValueError('Data base must contain only numbers to calculate the sum')
                
                result += item
        
        return result
    
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
        
        return std
    
    @staticmethod
    def read_csv(path: str, separator: str = ',', columns: list[str] | None = []) -> 'Database':
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
                    if item.isnumeric():
                        row.append(int(item))
                    elif item.replace('.', '', 1).isnumeric():
                        row.append(float(item))
                    else:
                        row.append(item)
                
                if columns is None:
                    columns = [str(i) for i in range(len(row))]
                    
                if len(columns) == 0:
                    columns = [str(item) for item in row]
                else:
                    values.append(row)
        
        if columns is None:
            columns = []
            
        return Database(columns, values)