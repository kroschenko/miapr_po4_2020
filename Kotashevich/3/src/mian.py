import numpy as np

def func(x):
    a = 0.3
    b = 0.3
    c = 0.08
    d = 0.3
    step = 0.3
    return a * np.cos(b * x * step) + c * np.sin(d * x * step)

def activation(x):
    return np.tanh(x)

def derivative(x):
    return 1 - (activation(x) ** 2)

def error(y,Y):
    return np.mean((y - Y) ** 2)

def training(inputs,predict,weights_hidden,weights_input):
    learning_rate = 0.5 # скорость обучения

    inputs_hidden = np.dot(weights_hidden,inputs)
    outputs_hidden = activation_mapper(inputs_hidden)

    inputs_input = np.dot(weights_input,outputs_hidden)
    outputs_input = activation(inputs_input)

    error_input = np.array([outputs_input[0] - predict])
    gradient_input = derivative(outputs_input[0])
    delta_input = error_input * gradient_input    
    weights_input -= learning_rate * np.dot(delta_input,outputs_hidden.reshape(1,len(outputs_hidden)))

    error_hidden = delta_input * weights_input
    gradient_hidden = derivative(outputs_hidden)
    delta_hidden = error_hidden * gradient_hidden
    weights_hidden -= learning_rate * np.dot(inputs.reshape(len(inputs),1),delta_hidden).T

    return weights_hidden,weights_input

def prediction(inputs,weights_hidden,weights_input):
    inputs_hidden = np.dot(weights_hidden,inputs)
    outputs_hidden = activation_mapper(inputs_hidden)

    inputs_input = np.dot(weights_input,outputs_hidden)
    outputs_input = activation(inputs_input)

    return outputs_input

activation_mapper = np.vectorize(activation)

learning = []
predictions = []
epoch = 0
epoch_maximum = 1000
error_minimum = 1e-5
n_input = 6
n_hidden = 2
n_train = 30
n_predict = 15
param = 0
w_hidden = np.random.normal(0.0,2 ** -0.5,(n_hidden,n_input))
w_input = np.random.normal(0.0,1,(1,n_hidden))

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
        w_hidden,w_input = training(np.array(sample),predict,w_hidden,w_input)
        inputs.append(np.array(sample))
        predicts.append(np.array(predict))
    error_learning = error(prediction(np.array(inputs).T,w_hidden,w_input),np.array(predicts))
    epoch += 1
    if error_learning <= error_minimum or epoch > epoch_maximum:
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

print("\nОБУЧЕНИЕ:")
for sample,predict in learning:
    output = prediction(sample,w_hidden,w_input)
    print("прогноз: {:<20} эталон: {:<30} отклонение: {:<20}".format(str(output),str(np.array(predict)),str(output - predict)))

print("\nПРОГНОЗИРОВАНИЕ:")
for sample,predict in predictions:
    output = prediction(sample,w_hidden,w_input)
    print("прогноз: {:<20} эталон: {:<30} отклонение: {:<20}".format(str(output),str(np.array(predict)),str(output - predict)))