import numpy
from math import sin
from math import cos
import random
import matplotlib.pyplot

# Функция принимает параметры:
# m1 - number neurons for learning
# m2 - number neurons for test
# a - parametr for etalon function
# b - parametr for etalon function
# c - parametr for etalon function
# d - parametr for etalon function
# step - parametr for etalon function
# inputs - number neurons
# hiddens - number neurons
# outputs - number neurons
# Ee - desired squared error
# alpha_ki - learning rate (inputs - hiddens)
# alpha_ij - learning rate (hiddens - outputs)
def lab3(m1, m2, a, b, c, d, step, inputs, hiddens, outputs, Ee, alpha_ki, alpha_ij):
    # Функция возвращает массив с эталонными значениями
    def get_etalons(n):
        etalons = numpy.zeros(n)
        for i in range(len(etalons)):
            x = step * i
            y = a * cos(b * x) + c * sin(d * x)
            etalons[i] = y
        return etalons

    etalons = get_etalons(m1 + m2)
    #print(etalons)

    # Функция возвращает массив с весами
    # Например, 8 нейронов в входном слое и 3 в скрытом, тогда возвратит массив 8х3
    # Например, 3 нейрона в скрытом слое и 1 нейрон в выходном слое, тогда возвратит массив 3х1
    def get_weights(leftNumberNeurons, rightNumberNeurons, leftRandomBorder, rightRandomBorder):
        weights = numpy.zeros((leftNumberNeurons, rightNumberNeurons))
        for i in range(leftNumberNeurons):
            for j in range(rightNumberNeurons):
                weights[i][j] = random.uniform(leftRandomBorder, rightRandomBorder)
        return weights

    weights_ki = get_weights(inputs, hiddens, -1, 1)
    #print(weights_ki)

    weights_ij = get_weights(hiddens, outputs, -1, 1)
    #print(weights_ij)

    # Функция возвращает массив с порогами
    # Например, 8 нейронов в входном слое и 3 в скрытом, тогда возвратит массив размером 3
    # Например, 3 нейрона в скрытом слое и 1 нейрон в выходном слое, тогда возвратит массив размером 1
    def get_thresholds(numberNeurons, leftRandomBorder, rightRandomBorder):
        tresholds = numpy.zeros(numberNeurons)
        for i in range(numberNeurons):
            tresholds[i] = random.uniform(leftRandomBorder, rightRandomBorder)
        return tresholds

    tresholds_i = get_thresholds(hiddens, -1, 1)
    #print(tresholds_i)

    tresholds_j = get_thresholds(outputs, -1, 1)
    #print(tresholds_j)

    valuesXforGraph = []
    valuesYforGraph = []
    eras = 0
    while 1:
        for q in range(m1 - inputs):
            # Si = x * wki - Ti
            x = etalons[q:(q+inputs)]
            S_i = x.dot(weights_ki) - tresholds_i
            #print(S_i)
            # yi = Sigm(Si)
            y_i = numpy.zeros(len(S_i))
            for i in range(len(S_i)):
                y_i[i] = 1. / (1. + numpy.exp( - S_i[i] )) # sigmoid func
            #print(y_i)
            # Sj = yi * wij - Tj
            S_j = y_i.dot(weights_ij) - tresholds_j
            #print(S_j)
            # yj = Linear(Sj) = Sj
            y_j = S_j
            #print(y_j)
            # jj = yj - e
            j_j = numpy.array([ y_j[i] - etalons[q + inputs + i] for i in range(len(y_j)) ])
            #print(j_j)
            # ji = sum [dF(Sj) * wij]
            dF_j = 1
            j_i = numpy.zeros(hiddens)
            for i in range(hiddens):
                for j in range(outputs):
                    j_i[i] += j_j[j] * dF_j * weights_ij[i][j]
            #print(j_i)
            # wij = wij - alpha * jj * dFj * yi
            for i in range(hiddens):
                for j in range(outputs):
                    weights_ij[i][j] -= alpha_ij * j_j[j] * dF_j * y_j[j]
            #print(weights_ij)
            # Tj = Tj + alpha * jj * dFj
            for j in range(outputs):
                tresholds_j += alpha_ij * j_j[j] * dF_j
            #print(tresholds_j)
            # wki = wki - alpha * ji * dFi * yi
            for k in range(inputs):
                for i in range(hiddens):
                    dFi = y_i[i] * (1 - y_i[i]) # derivative sigmoid func
                    weights_ki[k][i] -= alpha_ki * j_i[i] * dFi * y_i[i]
            #print(weights_ki)
            # Ti = Ti + alpha * ji * dFi
            for i in range(hiddens):
                dFi = y_i[i] * (1 - y_i[i]) # derivative sigmoid func
                tresholds_i += alpha_ki * j_i[i] * dFi
            #print(tresholds_i)
        E = 0
        for j in range(outputs):
            E += 1./2 * (y_j[j] - etalons[q + inputs + j]) ** 2
        valuesXforGraph.append(E)
        eras += 1
        valuesYforGraph.append(eras)
        print('eras: %8d\tE: %32.20f\r' % (eras, E), end = '')
        if E < Ee:
            print()
            matplotlib.pyplot.plot(valuesYforGraph, valuesXforGraph, 'g-o')
            matplotlib.pyplot.title('E(eras)')
            break

    print('after learning:')
    for q in range(m1):
        # Si = x * wki - Ti
        x = etalons[q:(q+inputs)]
        S_i = x.dot(weights_ki) - tresholds_i
        #print(S_i)
        # yi = Sigm(Si)
        y_i = numpy.zeros(len(S_i))
        for i in range(len(S_i)):
            y_i[i] = 1. / (1. + numpy.exp( - S_i[i] )) # sigmoid func
        #print(y_i)
        # Sj = yi * wij - Tj
        S_j = y_i.dot(weights_ij) - tresholds_j
        #print(S_j)
        # yj = Linear(Sj) = Sj
        y_j = S_j
        print('%8d\t%24.20f\t%24.20f\t%24.20f' % (
            q,
            etalons[q + inputs],
            y_j,
            (etalons[q + inputs] - y_j) ** 2)
        )

    print('test:')
    for q in range(m2 - inputs):
        # Si = x * wki - Ti
        x = etalons[(q + m1):(q + inputs + m1)]
        S_i = x.dot(weights_ki) - tresholds_i
        #print(S_i)
        # yi = Sigm(Si)
        y_i = numpy.zeros(len(S_i))
        for i in range(len(S_i)):
            y_i[i] = 1. / (1. + numpy.exp( - S_i[i] )) # sigmoid func
        #print(y_i)
        # Sj = yi * wij - Tj
        S_j = y_i.dot(weights_ij) - tresholds_j
        #print(S_j)
        # yj = Linear(Sj) = Sj
        y_j = S_j
        print('%8d\t%24.20f\t%24.20f\t%24.20f' % (
            q + m1,
            etalons[q + m1 + inputs],
            y_j,
            (etalons[q + m1 + inputs] - y_j) ** 2)
        )

    print('Eras: %d' % eras)

    matplotlib.pyplot.show()

random.seed(2)
lab3(
    30,     # m1 - number neurons for learning
    40,     # m2 - number neurons for test
    0.1,    # a - parametr for etalon function
    0.5,    # b - parametr for etalon function
    0.09,   # c - parametr for etalon function
    0.5,    # d - parametr for etalon function
    0.001,  # step - parametr for etalon function
    8,      # inputs - number neurons
    3,      # hiddens - number neurons
    1,      # outputs - number neurons
    1e-8,   # Ee - desired squared error
    0.001,  # alpha_ki - learning rate (inputs - hiddens)
    0.001   # alpha_ij - learning rate (hiddens - outputs)
)