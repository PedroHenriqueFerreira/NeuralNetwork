from database import Database

from neural_network import NeuralNetwork
from preprocessing import OneHotEncoder, Standardizer

db = Database.read_csv('digits.csv', columns=None)

y_encoder = OneHotEncoder()

db[-1] = y_encoder.fit_transform(db[-1])

X_standard_scaler = Standardizer()

db[:64] = X_standard_scaler.fit_transform(db[:64])

X = db[:64]
y = db[64:]

nn = NeuralNetwork(64, [32], 10, activation='tanh', output_activation='softmax', verbose=True)

nn.train(X.values, y.values)

for i in range(len(y.values)):
    print(y_encoder.inverse_transform(nn.predict(X.values[i])))