from sklearn import datasets, preprocessing
from neural_network import NeuralNetwork
from database import Database
import pandas as pd
from preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

digits = Database.read_csv('digits.csv', columns=None)

scaler = StandardScaler()

X = scaler.fit_transform(digits[:-1])
y = digits[-1]

# def one_hot_encoder(y: int) -> list[float]:
#     array = [0.0] * 10
    
#     array[y] = 1.0
#     return array

digits2 = datasets.load_digits()

print(y)
print('-------------')
print(digits2.target)

# print(X)
# print('----------')
# # print(digits2.data[0,:])
# print('----------')

# y = [one_hot_encoder(v) for v in digits.target]

# # X = X[:1000]
# # y = y[:1000]

# split_index = int(len(X) * 0.8)
# X_train, X_test = X[:split_index], X[split_index:]
# y_train, y_test = y[:split_index], y[split_index:]

# nn = NeuralNetwork(64, [60], 10, verbose=True)

# nn.train(X_train, y_train)

# correct_count = 0

# for X_item, y_item in zip(X_test, y_test):
#     predicted = [round(x) for x in nn.predict(X_item)]
#     expected = [round(y) for y in y_item]
    
#     if predicted == expected:
#         correct_count += 1
    
# print(f'Accuracy: {(correct_count / len(X_test)) * 100}%')