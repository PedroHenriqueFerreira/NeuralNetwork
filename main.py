from database import Database, LabelEncoder, NormalScaler

from sklearn.preprocessing import Binarizer as SKBinarizer

db = Database(['age', 'salary', 'status', 'weight'], [
    [0,1, 'single', 50],
    [1,1, 'married', 60],
    [2,2, 'single', 70],
    [3,3, 'married', 80],
    [4,4, 'single', 90],
])

transformer = NormalScaler()

db[0, 1, 3] = transformer.fit_transform(db[0, 1, 3])

print(db)