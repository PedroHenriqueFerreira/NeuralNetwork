from database import Database

class StandardScaler:
    def __init__(self):
        self.mean: list[float] = []
        self.std: list[float] = []    
    
    def fit(self, database: Database) -> None:
        ''' Fit the scaler to the data '''
        
        self.mean.clear()
        self.std.clear()   
        
        for column in database.columns:
            self.mean.append(database[column].mean())
            
            std = database[column].std()
            self.std.append(std if std != 0 else 1)
            
    def transform(self, database: Database) -> Database:
        ''' Transform the data using the scaler '''
        
        if len(self.mean) == 0 or len(self.std) == 0:
            raise ValueError('Scaler must be fitted before transforming data')
        
        if len(self.mean) != len(database.columns) or len(self.std) != len(database.columns):
            raise ValueError('Invalid database')
        
        result = Database()
        
        for i, column in enumerate(database.columns):
            result[column] = database[column].map(lambda x: (x - self.mean[i]) / self.std[i]).values
            
        return result
    
    def inverse_transform(self, database: Database) -> Database:
        ''' Inverse transform the data using the scaler '''
        
        if len(self.mean) == 0 or len(self.std) == 0:
            raise ValueError('Scaler must be fitted before transforming data')
        
        if len(self.mean) != len(database.columns) or len(self.std) != len(database.columns):
            raise ValueError('Invalid database')
        
        result = Database()
        
        for i, column in enumerate(database.columns):
            result[column] = database[column].map(lambda x: (x * self.std[i]) + self.mean[i]).values
            
        return result
    
    def fit_transform(self, database: Database) -> Database:
        ''' Fit and transform the data using the scaler '''
        
        self.fit(database)
        return self.transform(database)
    
class LabelEncoder:
    def __init__(self):
        self.classes: list[str] = []
        
    def fit(self, database: Database) -> None:
        ''' Fit the encoder to the data '''
        
        self.classes.clear()
        
        if len(database.columns) != 1:
            raise ValueError('Database must have a single column')
        
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
        self.categories: list[str] = []
        
    def fit(self, database: Database) -> None:
        ''' Fit the encoder to the data '''
        
        self.categories.clear()
        
        if len(database.columns) != 1:
            raise ValueError('Database must have a single column')
        
        self.categories.extend(database.unique())
    
    def transform(self, database: Database) -> Database:
        ''' Transform the data using the encoder '''
        
        if len(self.categories) == 0:
            raise ValueError('Encoder must be fitted before transforming data')
        
        if len(database.columns) != 1:
            raise ValueError('Database must have a single column')
        
        self.column = database.columns[0]
        
        result = Database()
        
        for category in self.categories:
            result[f'{self.column}_{category}'] = database.map(lambda x: int(x == category)).values
            
        return result
    
    def inverse_transform(self, database: Database) -> Database:
        ''' Inverse transform the data using the encoder '''
        
        if len(self.categories) == 0:
            raise ValueError('Encoder must be fitted before transforming data')
        
        if len(database.columns) != len(self.categories):
            raise ValueError('Invalid database')
        
        return Database([self.column], [[self.categories[row.index(1)]] for row in database.values])
    
    def fit_transform(self, database: Database) -> Database:
        ''' Fit and transform the data using the encoder '''
        
        self.fit(database)
        return self.transform(database)