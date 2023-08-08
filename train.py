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

nn = NeuralNetwork([60], verbose=True)

X_train, X_test = digits[:64].split(0.95)
y_train, y_test = digits[64:].split(0.95)

nn.fit(X_train, y_train)

print(nn.accuracy(X_test, y_test))

nn.to_json('digits/neural_network.json')

