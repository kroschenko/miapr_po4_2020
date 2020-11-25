import numpy as np
#import sys

def func(x):
    a, b, c, d, step = 0.2, 0.6, 0.05, 0.6, 0.1
    return a * np.cos(b * x * step) + c * np.sin(d * x * step)

def activation(x):
    return np.tanh(x)
    #e = 2.718
    #return 1 / (1 + e ** (-x))
    #return 1 / (1 + np.exp(-x))

def derivative(x):
    return 1 - (activation(x) ** 2)
    #return -((activation(x) ** 2) - 1)
    #return x * (1 - x)

def error(y,Y):
    return np.mean((y - Y) ** 2)

def adaptive(errors,outputs,inputs):
    return np.divide(np.sum(np.dot(np.square(errors),np.subtract(1,np.square(outputs)))),np.multiply(np.add(1,np.sum(np.square(inputs))),np.sum(np.dot(np.square(errors),np.square(np.subtract(1,np.square(outputs)))))))

def training(inputs,predict,weights_hidden,weights_input,learning_rate):
    inputs_hidden  = np.dot(weights_hidden,inputs)
    outputs_hidden = activation_mapper(inputs_hidden)

    inputs_input  = np.dot(weights_input,outputs_hidden)
    outputs_input = activation(inputs_input)

    error_input    = np.array([outputs_input[0] - predict])
    gradient_input = derivative(outputs_input[0])
    delta_input    = error_input * gradient_input    
    weights_input -= learning_rate * np.dot(delta_input,outputs_hidden.reshape(1,len(outputs_hidden)))

    #learning_rate = adaptive(error_input,outputs_hidden) # адаптивная скорость обучения

    error_hidden    = delta_input * weights_input
    gradient_hidden = derivative(outputs_hidden)
    delta_hidden    = error_hidden * gradient_hidden
    weights_hidden -= learning_rate * np.dot(inputs.reshape(len(inputs),1),delta_hidden).T

    learning_rate = adaptive(error_hidden,outputs_hidden,inputs) # адаптивная скорость обучения

    return weights_hidden,weights_input,learning_rate

def prediction(inputs,weights_hidden,weights_input):
    inputs_hidden  = np.dot(weights_hidden,inputs)
    outputs_hidden = activation_mapper(inputs_hidden)

    inputs_input  = np.dot(weights_input,outputs_hidden)
    outputs_input = activation(inputs_input)

    return outputs_input

activation_mapper = np.vectorize(activation)

learning      = []
predictions   = []
learning_rate = 0.5
epoch         = 0
#epoch_maximum = 1000
error_minimum = 1e-5  # минимальная ошибка
n_input       = 10    # количество входов
n_hidden      = 4     # количество элементов скрытого слоя
n_train       = 30    # размер выборки для обучения
n_predict     = 15    # размер выборки для прогназированния
param         = 0
w_hidden      = np.random.normal(0.0,2 ** -0.5,(n_hidden,n_input))
w_input       = np.random.normal(0.0,1,(1,n_hidden))
#w_hidden      = np.random.normal(0.0,0.1,(n_hidden,n_input))
#w_input       = np.random.normal(0.0,0.1,(1,n_hidden))

for i in range(n_train):
    inp, com = [], []
    for j in range(n_input):
        inp.append(func(param))
        param += 1
    com.append(inp)
    com.append(func(param))
    learning.append(tuple(com))

while True:
    inputs, predicts = [], []
    for sample,predict in learning:
        w_hidden,w_input,learning_rate = training(np.array(sample),predict,w_hidden,w_input,learning_rate)
        inputs.append(np.array(sample))
        predicts.append(np.array(predict))
    error_learning = error(prediction(np.array(inputs).T,w_hidden,w_input),np.array(predicts))
    epoch         += 1
    #sys.stdout.write("\rОшибка: {}, Эпохи: {}".format(str(error_learning),str(epoch)))
    if error_learning <= error_minimum:
        break

print("Ошибка: {}, Эпохи: {}".format(str(error_learning),str(epoch)))

for i in range(n_train,n_train + n_predict):
    inp, com = [], []
    for j in range(n_input):
        inp.append(func(param))
        param += 1
    com.append(inp)
    com.append(func(param))
    predictions.append(tuple(com))

print("\nРЕЗУЛЬТАТЫ ОБУЧЕНИЯ:")
for sample,predict in learning:
    output = prediction(sample,w_hidden,w_input)
    print("прогноз: {:<20} ожидаемый: {:<30} погрешность: {:<20}".format(str(output),str(np.array(predict)),str(output - predict)))

print("\nРЕЗУЛЬТАТЫ ПРОГНОЗИРОВАНИЯ:")
for sample,predict in predictions:
    output = prediction(sample,w_hidden,w_input)
    print("прогноз: {:<20} ожидаемый: {:<30} погрешность: {:<20}".format(str(output),str(np.array(predict)),str(output - predict)))
