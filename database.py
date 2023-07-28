from typing import Any, Callable, TypeVar, List

T = TypeVar('T')

class DataBase:
    ''' Data base class '''
    
    def __init__(self, columns: List[str], data: List[List[Any]]):
        ''' Create a new data base instance with the given columns and data '''
        
        if len(set(columns)) != len(columns):
            raise ValueError('Columns must be unique')
        
        if len(set([len(row) for row in data])) != 1:
            raise ValueError('All rows must have the same length')  
        
        if len(data) > 0 and len(columns) != len(data[0]):
            raise ValueError('Number of columns must be equal to the number of rows')
        
        self.columns = columns
        self.data = data
    
    def __str__(self):
        ''' Return a string representation of the data base '''
        
        columns_width: list[int] = []
        
        for i in range(len(self.columns)):
            width = max([len(str(row[i])) for row in [self.columns, *self.data]])
            
            columns_width.append(width)
        
        lines: list[str] = []
        
        for row in [self.columns, *self.data]:
            lines.append('  '.join([str(item).ljust(columns_width[i]) for i, item in enumerate(row)]))
            
        return '\n'.join(lines)
        
    def filter(self, columns: dict[str, Callable[[Any], bool]]) -> 'DataBase':
        ''' Filter the data base based on the given columns and functions '''
        
        indexes: dict[int, Callable[[Any], bool]] = {}
        
        for column in columns:
            if column not in self.columns:
                raise ValueError(f'Column {column} not found')
            
            indexes[self.columns.index(column)] = columns[column]
        
        data: list[list[Any]] = []
        
        for row in self.data:
            is_valid = True
            
            for index in indexes:
                if not indexes[index](row[index]):
                    is_valid = False
                    break
            
            if not is_valid:
                continue
            
            data.append(row.copy())
            
        return DataBase(self.columns.copy(), data)
        
    def select_column_data(self, column: str) -> list[Any]:
        ''' Select the data from a column '''
        
        if column not in self.columns:
            raise ValueError(f'Column {column} not found')
        
        index = self.columns.index(column)
        
        return [row[index] for row in self.data if row[index]]
    
    def select_columns(self, columns: list[str]) -> 'DataBase':
        ''' Select the data from the given columns '''
        
        indexes = [self.columns.index(column) for column in columns]
        
        data = []
        
        for row in self.data:            
            row = [row[index] for index in indexes]
            
            if len([item for item in row if not item]) > 0:
                continue
            
            data.append(row.copy())
        
        return DataBase(columns, data)
    
    def group_by(self, column: str) -> dict[str, 'DataBase']:
        ''' Group the data base by the given column '''
        
        if column not in self.columns:
            raise ValueError(f'Column {column} not found')
        
        index = self.columns.index(column)
        
        groups: dict[str, DataBase] = {}
        
        for row in self.data:
            if row[index] not in groups:
                groups[row[index]] = DataBase(self.columns.copy(), [])
                
            groups[row[index]].data.append(row.copy())
            
        return groups
        
    @staticmethod
    def read_csv(path: str, separator: str = ',') -> 'DataBase':
        ''' Read a csv file and return a data base '''
        
        columns: list[str] = []
        data: list[list[Any]] = []
        
        with open(path, 'r') as f:
            for line in f.readlines():
                items = line.strip().split(separator)

                row: list[Any] = []                

                for item in items:
                    if item == '':
                        row.append(None)
                    elif item.isnumeric():
                        row.append(int(item))
                    elif item.replace('.', '', 1).isnumeric():
                        row.append(float(item))
                    else:
                        row.append(item)
                
                if len(columns) == 0:
                    columns.extend(row)
                else:
                    data.append(row)
                    
        return DataBase(columns, data)