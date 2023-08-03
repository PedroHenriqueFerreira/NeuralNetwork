from typing import Any

from database.database import Database

class LabelEncoder:
    def __init__(self):
        self.categories: list[list[Any]] = []
        
    def fit(self, database: Database) -> None:
        ''' Fit the encoder to the data '''
        
        self.categories.clear()
        
        for column in database.columns:
            self.categories.append(database[column].unique())
    
    def transform(self, database: Database) -> Database:
        ''' Transform the data using the encoder '''
        
        if len(self.categories) == 0:
            raise ValueError('Encoder must be fitted before transforming data')
        
        if len(self.categories) != len(database.columns):
            raise ValueError('Invalid database')
        
        columns = database.columns[:]
        values: list[list[Any]] = []
        
        for row in database.values:
            values.append([self.categories[i].index(item) for i, item in enumerate(row)])
        
        return Database(columns, values)
    
    def inverse_transform(self, database: Database) -> Database:
        ''' Inverse transform the data using the encoder '''
        
        if len(self.categories) == 0:
            raise ValueError('Encoder must be fitted before transforming data')
        
        if len(self.categories) != len(database.columns):
            raise ValueError('Invalid database')
    
        columns = database.columns[:]
        values: list[list[Any]] = []
        
        for row in database.values:
            values.append([self.categories[i][item] for i, item in enumerate(row)])
    
        return Database(columns, values)
    
    def fit_transform(self, database: Database) -> Database:
        ''' Fit and transform the data using the encoder '''
        
        self.fit(database)
        return self.transform(database)
    
class OneHotEncoder:
    def __init__(self):
        self.columns: list[str] = []
        self.categories: list[list[Any]] = []
        
    def fit(self, database: Database) -> None:
        ''' Fit the encoder to the data '''
        
        self.columns.clear()
        self.categories.clear()
        
        for column in database.columns:
            self.columns.append(column)
            self.categories.append(database[column].unique())
    
    def transform(self, database: Database) -> Database:
        ''' Transform the data using the encoder '''
        
        if len(self.categories) == 0:
            raise ValueError('Encoder must be fitted before transforming data')
        
        if len(self.categories) != len(database.columns):
            raise ValueError('Invalid database')
        
        columns: list[str] = []
        
        for column, category in zip(self.columns, self.categories):
            if column not in database.columns:
                raise ValueError('Invalid database')
        
            columns.extend([f'{column}_{value}' for value in category])
        
        values: list[list[Any]] = []
        
        for row in database.values:
            values.append([])
            
            for item, category in zip(row, self.categories):
                values[-1].extend([int(item == value) for value in category])
        
        return Database(columns, values)
    
    def inverse_transform(self, database: Database) -> Database:
        ''' Inverse transform the data using the encoder '''
        
        if len(self.categories) == 0:
            raise ValueError('Encoder must be fitted before transforming data')
        
        if len(self.categories) != len(database.columns):
            raise ValueError('Invalid database')
        
        return Database()
        # return Database([self.column], [[self.classes[row.index(1)]] for row in database.values])
    
    def fit_transform(self, database: Database) -> Database:
        ''' Fit and transform the data using the encoder '''
        
        self.fit(database)
        return self.transform(database)
    
__all__ = ['LabelEncoder', 'OneHotEncoder']