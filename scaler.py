from database import Database

class Scaler:
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
            
            if std == 0:
                std = 1
            
            self.std.append(std)
            
    def transform(self, database: Database) -> Database:
        ''' Transform the data using the scaler '''
        
        if len(self.mean) == 0 or len(self.std) == 0:
            raise ValueError('Scaler must be fitted before transforming data')
        
        if len(self.mean) != len(database.columns) or len(self.std) != len(database.columns):
            raise ValueError('Scaler must be fitted to the same number of columns as the database')
        
        for i, column in enumerate(database.columns):
            database[column] = database[column].map(lambda x: (x - self.mean[i]) / self.std[i])
            
        return database
    
    def inverse_transform(self, database: Database) -> Database:
        ''' Inverse transform the data using the scaler '''
        
        if len(self.mean) == 0 or len(self.std) == 0:
            raise ValueError('Scaler must be fitted before transforming data')
        
        if len(self.mean) != len(database.columns) or len(self.std) != len(database.columns):
            raise ValueError('Scaler must be fitted to the same number of columns as the database')
        
        for i, column in enumerate(database.columns):
            database[column] = database[column].map(lambda x: (x * self.std[i]) + self.mean[i])
            
        return database
    
    def fit_transform(self, database: Database) -> Database:
        ''' Fit and transform the data using the scaler '''
        
        self.fit(database)
        return self.transform(database)