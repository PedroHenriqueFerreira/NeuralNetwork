from typing import Any

from database import Database

class Binarizer:
    ''' Binarize the data using a threshold '''

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def fit(self, database: Database) -> None:
        ''' Fit the binarizer to the data '''
        
        if database.map(lambda x: not isinstance(x, int | float)).sum() > 0:
            raise ValueError('Invalid database')

    def transform(self, database: Database) -> Database:
        ''' Transform the data using the binarizer '''
        
        return database.map(lambda x: int(x > self.threshold))
    
    def fit_transform(self, database: Database) -> Database:
        ''' Fit and transform the data using the binarizer '''
        
        self.fit(database)
        return self.transform(database)

class Normalizer:
    ''' Normalize the data using the L2 norm '''
    
    def __init__(self):
        self.norms: list[float] = []
    
    def fit(self, database: Database) -> None:
        ''' Fit the normalizer to the data '''
        
        if database.map(lambda x: not isinstance(x, int | float)).sum() > 0:
            raise ValueError('Invalid database')
        
        self.norms.clear()
        
        for row in database.values:
            norm = sum([item ** 2 for item in row]) ** 0.5
            
            self.norms.append(norm if norm != 0 else 1)

    def transform(self, database: Database) -> Database:
        ''' Transform the data using the normalizer '''
        
        if len(self.norms) == 0:
            raise ValueError('Normalizer must be fitted before transforming data')
        
        if len(self.norms) != len(database.values):
            raise ValueError('Invalid database')
        
        result = Database(database.columns[:])
        
        for i, row in enumerate(database.values):
            result.values.append([item / self.norms[i] for item in row])
            
        return result
    
    def inverse_transform(self, database: Database) -> Database:
        ''' Inverse transform the data using the normalizer '''
        
        if len(self.norms) == 0:
            raise ValueError('Normalizer must be fitted before transforming data')
        
        if len(self.norms) != len(database.values):
            raise ValueError('Invalid database')
        
        result = Database(database.columns[:])
        
        for i, row in enumerate(database.values):
            result.values.append([item * self.norms[i] for item in row])
            
        return result
    
    def fit_transform(self, database: Database) -> Database:
        ''' Fit and transform the data using the normalizer '''
        
        self.fit(database)
        return self.transform(database)

class Standardizer:
    ''' Standardize the data using the mean and standard deviation '''
    
    def __init__(self):
        self.mean: list[float] = []
        self.std: list[float] = []    
    
    def fit(self, database: Database) -> None:
        ''' Fit the standardizer to the data '''
        
        if database.map(lambda x: not isinstance(x, int | float)).sum() > 0:
            raise ValueError('Invalid database')
        
        self.mean.clear()
        self.std.clear()   
        
        for column in database.columns:
            std = database[column].std()
            
            self.mean.append(database[column].mean())
            self.std.append(std if std != 0 else 1)
            
    def transform(self, database: Database) -> Database:
        ''' Transform the data using the standardizer '''
        
        if len(self.mean) == 0 or len(self.std) == 0:
            raise ValueError('Scaler must be fitted before transforming data')
        
        if len(self.mean) != len(database.columns) or len(self.std) != len(database.columns):
            raise ValueError('Invalid database')
        
        result = Database()
        
        for i, column in enumerate(database.columns):
            result[column] = database[column].map(lambda x: (x - self.mean[i]) / self.std[i])
            
        return result
    
    def inverse_transform(self, database: Database) -> Database:
        ''' Inverse transform the data using the standardizer '''
        
        if len(self.mean) == 0 or len(self.std) == 0:
            raise ValueError('Scaler must be fitted before transforming data')
        
        if len(self.mean) != len(database.columns) or len(self.std) != len(database.columns):
            raise ValueError('Invalid database')
        
        result = Database()
        
        for i, column in enumerate(database.columns):
            result[column] = database[column].map(lambda x: (x * self.std[i]) + self.mean[i])
            
        return result
    
    def fit_transform(self, database: Database) -> Database:
        ''' Fit and transform the data using the standardizer '''
        
        self.fit(database)
        return self.transform(database)
    
class LabelEncoder:
    def __init__(self):
        self.classes: list[Any] = []
        
    def fit(self, database: Database) -> None:
        ''' Fit the encoder to the data '''
        
        if len(database.columns) != 1:
            raise ValueError('Database must have a single column')
        
        self.classes.clear()
        
        self.classes.extend(database.unique())
    
    def transform(self, database: Database) -> Database:
        ''' Transform the data using the encoder '''
        
        if len(self.classes) == 0:
            raise ValueError('Encoder must be fitted before transforming data')
        
        if len(database.columns) != 1:
            raise ValueError('Database must have a single column')
        
        return database.map(lambda x: self.classes.index(x))
    
    def inverse_transform(self, database: Database) -> Database:
        ''' Inverse transform the data using the encoder '''
        
        if len(self.classes) == 0:
            raise ValueError('Encoder must be fitted before transforming data')
        
        if len(database.columns) != 1:
            raise ValueError('Database must have a single column')
    
        return database.map(lambda x: self.classes[x])
    
    def fit_transform(self, database: Database) -> Database:
        ''' Fit and transform the data using the encoder '''
        
        self.fit(database)
        return self.transform(database)
    
class OneHotEncoder:
    def __init__(self):
        self.column = ''
        self.classes: list[Any] = []
        
    def fit(self, database: Database) -> None:
        ''' Fit the encoder to the data '''
        
        if len(database.columns) != 1:
            raise ValueError('Database must have a single column')
        
        self.classes.clear()
        
        self.classes.extend(database.unique())
    
    def transform(self, database: Database) -> Database:
        ''' Transform the data using the encoder '''
        
        if len(self.classes) == 0:
            raise ValueError('Encoder must be fitted before transforming data')
        
        if len(database.columns) != 1:
            raise ValueError('Database must have a single column')
        
        self.column = database.columns[0]
        
        result = Database()
        
        for class_ in self.classes:
            result[f'{self.column}_{class_}'] = database.map(lambda x: int(x == class_)).values
            
        return result
    
    def inverse_transform(self, database: Database) -> Database:
        ''' Inverse transform the data using the encoder '''
        
        if len(self.classes) == 0:
            raise ValueError('Encoder must be fitted before transforming data')
        
        if len(database.columns) != len(self.classes):
            raise ValueError('Invalid database')
        
        return Database([self.column], [[self.classes[row.index(1)]] for row in database.values])
    
    def fit_transform(self, database: Database) -> Database:
        ''' Fit and transform the data using the encoder '''
        
        self.fit(database)
        return self.transform(database)
    