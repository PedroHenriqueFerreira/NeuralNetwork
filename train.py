from database import Database

from database.scalers import StandardScaler
from database.encoders import OneHotEncoder

from neural_network import NeuralNetwork

digits = Database.from_csv('digits/dataset.csv')

scaler = StandardScaler()
encoder = OneHotEncoder()

digits[:-1] = scaler.fit_transform(digits[:-1])
digits[-1] = encoder.fit_transform(digits[-1])

scaler.to_json('digits/X_scaler.json')
encoder.to_json('digits/y_encoder.json')

nn = NeuralNetwork([50, 50], verbose=True)

X_train = digits[:64]
y_train = digits[64:]

nn.fit(X_train, y_train)

nn.to_json('digits/neural_network.json')

