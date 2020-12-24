import numpy as np

#ФУНКЦИИ

def rel_func(x):
    return np.where(x <= 0, 0, 1)
    return np.maximum(0, x)

def drel_func(x):
    return np.where(x <= 0, 0, 1)

def sig_func(x):
    return np.tanh(x)

def dsig_func(x):
    return 1 - (sig_func(x) ** 2)

def error(y, Y):
    yy = y.ravel()
    YY = Y.ravel()
    return np.mean((yy.reshape(1, len(yy)) - YY.reshape(1, len(YY))) ** 2)

def adaptive(errors, outputs, inputs):
    return np.divide(np.sum(np.dot(np.square(errors), np.subtract(1, np.square(outputs)))),
                     np.multiply(np.add(1, np.sum(np.square(inputs))),
                                 np.sum(np.dot(np.square(errors), np.square(np.subtract(1, np.square(outputs)))))))

def training(inputs, predict, weights_hidden, weights_input, learning_rate):
    inputs_hidden = np.dot(weights_hidden, inputs)
    outputs_hidden = sigmoid_mapper(inputs_hidden)


    inputs_input = np.dot(weights_input, outputs_hidden)
    outputs_input = sigmoid_mapper(inputs_input)


    error_input = np.subtract(outputs_input, predict)
    gradient_input = dsig_func(outputs_input)
    delta_input = error_input * gradient_input
    for w, d in zip(weights_input, delta_input):
        ww, dd = [], []
        ww = w.reshape(1, len(w))
        dd.append(d)
        ww -= learning_rate * np.dot(dd, outputs_hidden.reshape(1, len(outputs_hidden)))

    for w, d in zip(weights_input, delta_input):
        ww, dd = [], []
        ww = w.reshape(1, len(w))
        dd.append(d)
        error_hidden = dd * ww
    gradient_hidden = dsig_func(outputs_hidden)
    delta_hidden = error_hidden * gradient_hidden
    weights_hidden -= learning_rate * np.dot(inputs.reshape(len(inputs), 1), delta_hidden).T
    return weights_hidden, weights_input, learning_rate

def prediction(inputs, weights_hidden, weights_input):
    inputs_hidden = np.dot(weights_hidden, inputs)
    outputs_hidden = sigmoid_mapper(inputs_hidden)


    inputs_input = np.dot(weights_input, outputs_hidden)
    outputs_input = sigmoid_mapper(inputs_input)

    return outputs_input


#ДАННЫЕ

sigmoid_mapper = np.vectorize(sig_func)
relu_mapper = np.vectorize(rel_func)

learning = []
predictions = []
learning_rate = 0.5
epoch = 0
epoch_maximum = 15000
error_minimum = 1e-6
n_input = 20
n_hidden = 10
n_output = 3
w_hidden = np.random.normal(0.0, 2 ** -0.5, (n_hidden, n_input))
w_input = np.random.normal(0.0, 1, (n_output, n_hidden))


#ДАННЫЕ ВЕКТОРОВ

vectors = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1,1, 1, 1,1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1]])
codes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

for vector, code in zip(vectors, codes):
    com = []
    com.append(vector)
    com.append(code)
    learning.append(tuple(com))

while True:
    inputs, predicts = [], []
    for sample, predict in learning:
        w_hidden, w_input, learning_rate = training(np.array(sample), np.array(predict), w_hidden, w_input,
                                                    learning_rate)
        inputs.append(np.array(sample))
        predicts.append(np.array(predict))
    error_learning = error(prediction(np.array(inputs).T, w_hidden, w_input), np.array(predicts))
    epoch += 1
    if error_learning <= error_minimum or epoch > epoch_maximum:
        break

print("ЭПОХИ: ", epoch)

print("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ:\n")
for sample, predict in learning:
    output = prediction(sample, w_hidden, w_input)
    print("ПРОГНОЗ  : {:<30}, ОЖИДАЕМЫЙ ПРОГНОЗ: {:<30}\n".format(str(output), str(np.array(predict))))

vvectors = np.array([[1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
ccodes = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]])
for vector, code in zip(vvectors, ccodes):
    com = []
    com.append(vector)
    com.append(code)
    predictions.append(tuple(com))

print("\nРЕЗУЛЬТАТЫ ПРОГНОЗИРОВАНИЯ:")
for sample, predict in predictions:
    output = prediction(sample, w_hidden, w_input)
    print("ПРОГНОЗ  : {:<30}, ОЖИДАЕМЫЙ ПРОГНОЗ: {:<30}\n".format(str(output), str(np.array(predict))))

predictions = []
vvectors = np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                     [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                     [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0],
                     [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0],
                     [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0]])
ccodes = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]])
for vector, code in zip(vvectors, ccodes):
    com = []
    com.append(vector)
    com.append(code)
    predictions.append(tuple(com))

for sample, predict in predictions:
    output = prediction(sample, w_hidden, w_input)
    print("ПРОГНОЗ  : {:<30}, ОЖИДАЕМЫЙ ПРОГНОЗ: {:<30}\n".format(str(output), str(np.array(predict))))

predictions = []
vvectors = np.array([[1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1],
                     [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1],
                     [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
                     [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
                     [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1]])
ccodes = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]])
for vector, code in zip(vvectors, ccodes):
    com = []
    com.append(vector)
    com.append(code)
    predictions.append(tuple(com))

for sample, predict in predictions:
    output = prediction(sample, w_hidden, w_input)
    print("ПРОГНОЗ  : {:<30}, ОЖИДАЕМЫЙ ПРОГНОЗ: {:<30}\n".format(str(output), str(np.array(predict))))

