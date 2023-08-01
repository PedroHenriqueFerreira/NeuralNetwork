from database import Database
from preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from neural_network import NeuralNetwork

scaler = StandardScaler()
status_hot_encoder = OneHotEncoder()
status_label_encoder = LabelEncoder()

db = Database.read_csv('test.csv')

db['status'] = status_label_encoder.fit_transform(db['status'])
db['status'] = status_hot_encoder.fit_transform(db['status'])

db = scaler.fit_transform(db)

X = db[:-1].values
y = db[-1].values

nn = NeuralNetwork(4, [3, 3], 1, verbose=True, output_activation='identity')

nn.train(X, y)

for i, row in enumerate(X):
    print(nn.predict(row), y[i])

