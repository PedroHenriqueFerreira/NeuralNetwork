from database import Database
from preprocessing import LabelEncoder, OneHotEncoder, Standardizer
from neural_network import NeuralNetwork

standardizer = Standardizer()
status_hot_encoder = OneHotEncoder()

db = Database.read_csv('test.csv')

db['status'] = status_hot_encoder.fit_transform(db['status'])

db[:-1] = standardizer.fit_transform(db[:-1])

print(db)

# X = db[:-1].values
# y = db[-1].values

# nn = NeuralNetwork(4, [3, 3], 1, verbose=True)

# nn.train(X, y)

# for i, row in enumerate(X):
#     print(nn.predict(row), y[i])

