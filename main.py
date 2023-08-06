from database import Database

from database.scalers import StandardScaler
from database.encoders import OneHotEncoder

from neural_network import NeuralNetwork

digits = Database.from_csv('digits/dataset.csv', columns=None)

scaler = StandardScaler()
encoder = OneHotEncoder()

digits[:-1] = scaler.fit_transform(digits[:-1])
digits[-1] = encoder.fit_transform(digits[-1])

scaler.to_json('digits/X_scaler.json')
encoder.to_json('digits/y_encoder.json')

nn = NeuralNetwork([60], loss='mse', verbose=True)

X_train = digits[:64]
y_train = digits[64:]

nn.fit(X_train.values, y_train.values)

print(f'Accuracy: {nn.accuracy(X_train.values, y_train.values)}')
    
nn.to_json('digits/neural_network.json')

