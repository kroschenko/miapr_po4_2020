import numpy as np
import sys
import math

def func(x):
    a = 0.2
    b = 0.4
    c = 0.09
    d = 0.4
    return a * math.cos(b * x) + c * math.sin(d * x)

class Network:
    def __init__(self, learning_rate = 0.25):
        self.weights_0_1 = np.random.normal(0.0, 2 ** -0.5, (2, 6))
        self.weights_1_2 = np.random.normal(0.0, 1, (1, 2))

        self.tanh_mapper = np.vectorize(self.tanh)
        self.learning_rate = np.array([learning_rate])

    def tanh(self, x):
        return np.tanh(x)

    def predict(self, inputs):
        inputs_1 = np.dot(self.weights_0_1, inputs) 
        outputs_1 = self.tanh_mapper(inputs_1)

        inputs_2 = np.dot(self.weights_1_2, outputs_1) 
        outputs_2 = self.tanh(inputs_2)
        return outputs_2

    def train(self, inputs, expected_predict):
        inputs_1 = np.dot(self.weights_0_1, inputs) 
        outputs_1 = self.tanh_mapper(inputs_1)

        inputs_2 = np.dot(self.weights_1_2, outputs_1) 
        outputs_2 = self.tanh(inputs_2)
        actual_predict = outputs_2[0]

        error_layer_2 = np.array([actual_predict - expected_predict])
        gradient_layer_2 = -(self.tanh(actual_predict) ** 2) - 1
        weights_delta_layer_2 = error_layer_2 * gradient_layer_2    
        self.weights_1_2 -= (np.dot(weights_delta_layer_2, outputs_1.reshape(1, len(outputs_1)))) * self.learning_rate

        self.learning_rate = (np.sum(np.square(error_layer_2) * (np.ones((1, 2)) - np.square(outputs_2))))/((1 + np.sum(np.square(outputs_1)))*(np.sum(np.square(error_layer_2)*np.square(np.ones((1, 2)) - np.square(outputs_2)))))

        error_layer_1 = weights_delta_layer_2 * self.weights_1_2
        gradient_layer_1 = -(self.tanh(outputs_1) ** 2) - 1
        weights_delta_layer_1 = error_layer_1 * gradient_layer_1
        self.weights_0_1 -= np.dot(inputs.reshape(len(inputs), 1), weights_delta_layer_1).T * self.learning_rate
       

        


def MSE(y, Y):
    return np.mean((y - Y) ** 2)


step = 0.1
counter = 0
train = []
for i in range(-15, 15):
    combol = []
    inputs = []
    for j in range(6):
        x = counter * step
        inputs.append(func(x))
        counter += 1
    combol.append(inputs)
    x = counter * step
    combol.append(func(x))
    combo = tuple(combol)
    train.append(combo)


learning_rate = 0.05
network = Network(learning_rate)

losses = {'train':[], 'validation':[]}

Emin = 1e-6
train_loss = 0

epoch = 0
while True:
    inputs = []
    correct_predictions = []
    for input_stat, correct_predict in train:
        network.train(np.array(input_stat), correct_predict)
        inputs.append(np.array(input_stat))
        correct_predictions.append(np.array(correct_predict))

    train_loss = MSE(network.predict(np.array(inputs).T), np.array(correct_predictions))
    epoch += 1
    if train_loss <= Emin:
        break

print("epochs: {}, training loss: {}".format(
    str(epoch),
    str(train_loss)
    ))

print("\nРЕЗУЛЬТАТЫ ОБУЧЕНИЯ:")
for input_stat, correct_predict in train:
    print("the prediction is: {}, expected: {}, mistake: {}".format(
        str(network.predict(input_stat)),
        str(correct_predict),
        str(network.predict(input_stat) - correct_predict)
        ))

predict = []
for i in range(30, 45):
    combol = []
    inputs = []
    for j in range(6):
        x = counter * step
        inputs.append(func(x))
        counter += 1
    combol.append(inputs)
    x = counter * step
    combol.append(func(x))
    combo = tuple(combol)
    predict.append(combo)

print("\nРЕЗУЛЬТАТЫ ПРОГНОЗИРОВАНИЯ")
for input_stat, correct_predict in predict:
    print("the prediction is: {}, expected: {}, mistake: {}".format(
        str(network.predict(input_stat)),
        str(correct_predict),
        str(network.predict(input_stat) - correct_predict)
        ))

