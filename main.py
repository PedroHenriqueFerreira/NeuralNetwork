from sklearn import datasets, preprocessing
from neural_network import NeuralNetwork

def one_hot_encoder(y: int) -> list[float]:
    array = [0.0] * 10
    
    array[y] = 1.0
    return array

digits = datasets.load_digits()

X: list[list[float]] = preprocessing.scale(digits.data.astype(float))
y = [one_hot_encoder(v) for v in digits.target]

# X = X[:1000]
# y = y[:1000]

split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

nn = NeuralNetwork(64, [60], 10, verbose=True)

nn.train(X_train, y_train)

correct_count = 0

for X_item, y_item in zip(X_test, y_test):
    predicted = [round(x) for x in nn.predict(X_item)]
    expected = [round(y) for y in y_item]
    
    if predicted == expected:
        correct_count += 1
    
print(f'Accuracy: {(correct_count / len(X_test)) * 100}%')

# nn = NeuralNetwork(2, [2], 1, verbose=True, output_activation='sigmoid', tol=1e-6, max_iter=50000)

# X: list[list[float]] = [[0, 0], [0, 1], [1, 0], [1, 1]]
# y: list[list[float]] = [[0], [0], [0], [1]]

# nn.train(X, y)

# print()

# for X_item, y_item in zip(X, y):
#     predicted = nn.predict(X_item)
#     expected = y_item
    
#     print(f'Predicted: {predicted} | Expected: {expected}')