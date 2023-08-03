from database import Database

from database.scalers import StandardScaler
from database.encoders import OneHotEncoder

from neural_network import NeuralNetwork

digits = Database.from_csv('digits.csv', columns=None)

scaler = StandardScaler()
encoder = OneHotEncoder()

digits[:-1] = scaler.fit_transform(digits[:-1])
digits[-1] = encoder.fit_transform(digits[-1])

nn = NeuralNetwork([60], verbose=True, activation='tanh', output_activation='softmax')

X_train, X_test = digits[:64].split(0.8)
y_train, y_test = digits[64:].split(0.8)

nn.fit(X_train.values, y_train.values)

predict = nn.predict(X_test.values)
accuracy = sum(predict[i] == y_test.values[i] for i in range(len(predict))) / len(predict)
    
print(f'Accuracy: {accuracy * 100}%')