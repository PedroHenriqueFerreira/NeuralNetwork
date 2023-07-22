from sklearn.neural_network import MLPClassifier, MLPRegressor

from neural_network import NeuralNetwork

nn = NeuralNetwork(2, [2], 1, activation='relu', max_iter=50000, learning_rate=0.001)

nn.train([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], [
    [0],
    [1],
    [1],
    [0]
])

print(nn.predict([0, 0]))
print(nn.predict([0, 1]))
print(nn.predict([1, 0]))
print(nn.predict([1, 1]))