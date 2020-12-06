import numpy as np

def sigmoid(x):
    return np.tanh(x)

def dsigmoid(x):
    return 1 - (sigmoid(x) ** 2)

# Среднеквадратична ошибка
def error(y,Y):
    yy = y.ravel()
    YY = Y.ravel()
    return np.mean((yy.reshape(1,len(yy)) - YY.reshape(1,len(YY))) ** 2)

def adaptive(errors,outputs,inputs):
    return np.divide(np.sum(np.dot(np.square(errors),np.subtract(1,np.square(outputs)))),np.multiply(np.add(1,np.sum(np.square(inputs))),np.sum(np.dot(np.square(errors),np.square(np.subtract(1,np.square(outputs)))))))

def training(inputs,predict,weights_hidden,weights_input,learning_rate):
    
    # Выход скрытого слоя
    inputs_hidden  = np.dot(weights_hidden,inputs)
    outputs_hidden = sigmoid_mapper(inputs_hidden)

    # Выход выходного слоя
    inputs_input  = np.dot(weights_input,outputs_hidden)
    outputs_input = sigmoid_mapper(inputs_input)

    # Ошибка выходного слоя
    error_input    = np.subtract(outputs_input,predict)
    # Градиент выходного слоя
    gradient_input = dsigmoid(outputs_input)

    delta_input    = error_input * gradient_input
    for w,d in zip(weights_input,delta_input):
        # Корректируем выходные веса 
        ww, dd = [], []
        ww = w.reshape(1,len(w))
        dd.append(d)
        ww -= learning_rate * np.dot(dd,outputs_hidden.reshape(1,len(outputs_hidden)))

    for w,d in zip(weights_input,delta_input):
        # Ошибка скрытого слоя
        ww, dd = [], []
        ww = w.reshape(1,len(w))
        dd.append(d)
        error_hidden = dd * ww

    # Корректируем скрытые веса
    gradient_hidden = dsigmoid(outputs_hidden)
    delta_hidden    = error_hidden * gradient_hidden
    weights_hidden -= learning_rate * np.dot(inputs.reshape(len(inputs),1),delta_hidden).T

    return weights_hidden,weights_input,learning_rate

def prediction(inputs,weights_hidden,weights_input):
    inputs_hidden  = np.dot(weights_hidden,inputs)
    outputs_hidden = sigmoid_mapper(inputs_hidden)

    inputs_input  = np.dot(weights_input,outputs_hidden)
    outputs_input = sigmoid_mapper(inputs_input)

    return outputs_input

sigmoid_mapper = np.vectorize(sigmoid)

learning      = []
predictions   = []
learning_rate = 0.5
epoch         = 0
epoch_maximum = 15000
error_minimum = 1e-6  # минимальная ошибка
n_input       = 20    # количество входов
n_hidden      = 10    # количество элементов скрытого слоя
n_output      = 3     # количество выходов
w_hidden      = np.random.normal(0.0,2 ** -0.5,(n_hidden,n_input))
w_input       = np.random.normal(0.0,1,(n_output,n_hidden))

vectors = np.array([[1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],
                    [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],
                    [1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1]])

codes   = np.array([[1,0,0],[0,1,0],[0,0,1]])

for vector, code in zip(vectors, codes):
    com = []
    com.append(vector)
    com.append(code)
    learning.append(tuple(com))

while True:
    inputs, predicts = [], []
    for sample,predict in learning:
        w_hidden,w_input,learning_rate = training(np.array(sample),np.array(predict),w_hidden,w_input,learning_rate)
        inputs.append(np.array(sample))
        predicts.append(np.array(predict))
    error_learning = error(prediction(np.array(inputs).T,w_hidden,w_input),np.array(predicts))
    epoch         += 1
    if error_learning <= error_minimum or epoch > epoch_maximum:
        break


print("\nРЕЗУЛЬТАТЫ ОБУЧЕНИЯ:")
for sample,predict in learning:
    output = prediction(sample,w_hidden,w_input)
    print("прогноз  : {:<30}\nожидаемый: {:<30}\n".format(str(output),str(np.array(predict))))

vvectors = np.array([[1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],
                     [1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,0],
                     [1,0,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,0,1,1],
                     [1,1,1,1,0,0,0,0,1,1,0,1,0,0,0,0,1,1,1,1],
                     [1,1,1,1,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1]])

ccodes   = np.array([[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0]])

for vector, code in zip(vvectors, ccodes):
    com = []
    com.append(vector)
    com.append(code)
    predictions.append(tuple(com))

print("\nРЕЗУЛЬТАТЫ ПРОГНОЗИРОВАНИЯ:")
for sample,predict in predictions:
    output = prediction(sample,w_hidden,w_input)
    print("прогноз  : {:<30}\nожидаемый: {:<30}\n".format(str(output),str(np.array(predict))))

predictions = []
vvectors = np.array([[1,0,1,0,1,0,1,0,1,1,1,0,1,0,1,0,1,0,1,0],
                     [1,1,1,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,1,0],
                     [1,0,1,0,1,0,1,0,1,0,1,1,1,0,1,0,1,0,0,0],
                     [1,0,1,0,1,0,1,0,1,1,1,0,0,0,1,0,1,0,1,0],
                     [1,0,1,0,1,0,1,0,0,0,1,0,1,1,1,0,1,0,1,0]])

ccodes   = np.array([[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0]])
for vector, code in zip(vvectors, ccodes):
    com = []
    com.append(vector)
    com.append(code)
    predictions.append(tuple(com))

for sample,predict in predictions:
    output = prediction(sample,w_hidden,w_input)
    print("прогноз  : {:<30}\nожидаемый: {:<30}\n".format(str(output),str(np.array(predict))))

predictions = []
vvectors = np.array([[1,1,1,0,1,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1],
                     [1,1,1,0,0,0,1,1,0,0,0,0,1,1,1,0,0,0,1,1],
                     [1,1,1,0,0,0,1,1,1,0,1,0,1,1,1,0,0,0,1,1],
                     [0,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1],
                     [1,1,1,0,0,0,1,1,1,0,0,1,1,1,1,0,0,0,1,1]])
ccodes   = np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]])
for vector, code in zip(vvectors, ccodes):
    com = []
    com.append(vector)
    com.append(code)
    predictions.append(tuple(com))

for sample,predict in predictions:
    output = prediction(sample,w_hidden,w_input)
    print("прогноз  : {:<30}\nожидаемый: {:<30}\n".format(str(output),str(np.array(predict))))


