from sklearn import datasets, preprocessing
from neural_network import NeuralNetwork

def one_hot_encoder(y: int) -> list[float]:
    array = [0.0] * 10
    
    array[y] = 1.0
    return array

digits = datasets.load_digits()

X: list[list[float]] = preprocessing.scale(digits.data.astype(float))
y = [one_hot_encoder(v) for v in digits.target]

# X = X[:100]
# y = y[:100]

split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

nn = NeuralNetwork(64, [37, 37], 10)

nn.train(X_train, y_train)

for i in range(len(X_test)):
    print(f'PREDICT: {[x for x in nn.predict(X_test[i])]} | EXPECTED: {[round(y) for y in y_test[i]]}')