import numpy
import random
import matplotlib.pyplot

class lab5:
    # Функция принимает параметры:
    # variant - number neurons for learning
    # Ee - desired squared error
    # alpha_ki - learning rate (inputs - hiddens)
    # alpha_ij - learning rate (hiddens - outputs)
    def __init__(self, variant, maxIterations, Ee, alpha_ki, alpha_ij):
        print('\nObject lab5 are created\n')

        self.variant = variant
        self.maxIterations = maxIterations
        self.Ee = Ee
        self.alpha_ki = alpha_ki
        self.alpha_ij = alpha_ij

        self.etalons = self.get_etalons(self.variant)
        #print(self.etalons)

        self.inputs = len(self.etalons[0])
        #print(inputs)

        self.hiddens = self.inputs # взять столько же?

        self.outputs = len(self.etalons)
        #print(outputs)

        self.weights_ki = self.get_weights(self.inputs, self.hiddens, -1, 1)
        #print(weights_ki)

        self.weights_ij = self.get_weights(self.hiddens, self.outputs, -1, 1)
        #print(weights_ij)

        self.tresholds_i = self.get_thresholds(self.hiddens, -1, 1)
        #print(tresholds_i)

        self.tresholds_j = self.get_thresholds(self.outputs, -1, 1)
        #print(tresholds_j)

        self.valuesXforGraph = []
        self.valuesYforGraph = []

    def get_etalons(self, variant):
        variant %= 11

        Vector_1 = numpy.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0])
        Vector_2 = numpy.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
        Vector_3 = numpy.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])
        Vector_4 = numpy.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
        Vector_5 = numpy.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        Vector_6 = numpy.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        Vector_7 = numpy.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        Vector_8 = numpy.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1])

        if variant == 1:
            return numpy.array([Vector_1, Vector_6, Vector_8])

        elif variant == 2:
            return numpy.array([Vector_2, Vector_1, Vector_8])

        elif variant == 3:
            return numpy.array([Vector_3, Vector_2, Vector_8])

        elif variant == 4:
            return numpy.array([Vector_4, Vector_3, Vector_8])

        elif variant == 5:
            return numpy.array([Vector_5, Vector_4, Vector_8])

        elif variant == 6:
            return numpy.array([Vector_6, Vector_5, Vector_8])

        elif variant == 7:
            return numpy.array([Vector_7, Vector_6, Vector_8])

        elif variant == 8:
            return numpy.array([Vector_1, Vector_3, Vector_8])

        elif variant == 9:
            return numpy.array([Vector_2, Vector_4, Vector_8])

        elif variant == 10:
            return numpy.array([Vector_3, Vector_5, Vector_8])

        elif variant == 0: #11
            return numpy.array([Vector_4, Vector_6, Vector_8])

        else:
            return numpy.array([[], [], []])

    # Функция возвращает массив с весами
    # Например, 8 нейронов в входном слое и 3 в скрытом, тогда возвратит массив 8х3
    # Например, 3 нейрона в скрытом слое и 1 нейрон в выходном слое, тогда возвратит массив 3х1
    def get_weights(self, leftNumberNeurons, rightNumberNeurons, leftRandomBorder, rightRandomBorder):
        weights = numpy.zeros((leftNumberNeurons, rightNumberNeurons))
        for i in range(leftNumberNeurons):
            for j in range(rightNumberNeurons):
                weights[i][j] = random.uniform(leftRandomBorder, rightRandomBorder)
        return weights

    # Функция возвращает массив с порогами
    # Например, 8 нейронов в входном слое и 3 в скрытом, тогда возвратит массив размером 3
    # Например, 3 нейрона в скрытом слое и 1 нейрон в выходном слое, тогда возвратит массив размером 1
    def get_thresholds(self, numberNeurons, leftRandomBorder, rightRandomBorder):
        tresholds = numpy.zeros(numberNeurons)
        for i in range(numberNeurons):
            tresholds[i] = random.uniform(leftRandomBorder, rightRandomBorder)
        return tresholds

    def print_graph(self):
        print('\nFunction lab5 :: print_graph are start\n')

        matplotlib.pyplot.plot(self.valuesYforGraph, self.valuesXforGraph, 'g-o')
        matplotlib.pyplot.title('E(eras)')
        matplotlib.pyplot.show()

        print('\nFunction lab5 :: print_graph are end\n')

    def learning(self):
        print('\nFunction lab5 :: learning are start\n')

        self.valuesXforGraph = []
        self.valuesYforGraph = []
        eras = 0
        while 1:
            sum1 = 0
            sum2 = 0
            sum3 = 0
            for etalon in self.etalons:
                # Взвешенная сумма для скрытого слоя
                # Si = x * wki - Ti
                x = etalon
                S_i = x.dot(self.weights_ki) - self.tresholds_i
                #print(S_i)

                # Выходные значения для скрытого слоя
                # yi = Sigm(Si)
                y_i = numpy.zeros(len(S_i))
                for i in range(len(S_i)):
                    y_i[i] = 1. / (1. + numpy.exp( - S_i[i] )) # sigmoid func
                #print(y_i)

                # Взвешенная сумма для выходного слоя
                # Sj = yi * wij - Tj
                S_j = y_i.dot(self.weights_ij) - self.tresholds_j
                #print(S_j)

                # Выходные значения для выходного слоя
                # yj = Linear(Sj) = Sj
                y_j = S_j
                #print(y_j)

                # Обратная ошибка для выходного слоя
                # jj = yj - e
                j_j = numpy.array([ y_j[i] - etalon[i] for i in range(len(y_j)) ])
                #print(j_j)

                # Обратная ошибка для скрытого слоя
                # ji = sum [dF(Sj) * wij]
                dF_j = 1
                j_i = numpy.zeros(self.hiddens)
                for i in range(self.hiddens):
                    for j in range(self.outputs):
                        j_i[i] += j_j[j] * dF_j * self.weights_ij[i][j]
                #print(j_i)

                # # Первая сумма для адаптивного шага обучения
                # for j in range(outputs):
                #     sum1 += j_j[j]**2 * (1 - y_j[j]**2)
                # #print(sum1)

                # # Вторая сумма для адаптивного шага обучения
                # for i in range(hiddens):
                #     sum2 += y_i[i]**2
                # #print(sum2)

                # # Третья сумма для адаптивного шага обучения
                # for j in range(outputs):
                #     sum3 += j_j[j]**2 * y_j[j]**2 * (1 - y_j[j])**2
                # #print(sum3)

                # # Адаптивный шаг обучения
                # alpha_ki = 4 * sum1 / ( (1 + sum2) * sum3 )

                # Веса от скрытого слоя к выходному
                # wij = wij - alpha * jj * dFj * yi
                for i in range(self.hiddens):
                    for j in range(self.outputs):
                        self.weights_ij[i][j] -= self.alpha_ij * j_j[j] * dF_j * y_j[j]
                #print(weights_ij)

                # Пороги выходного слоя
                # Tj = Tj + alpha * jj * dFj
                for j in range(self.outputs):
                    self.tresholds_j += self.alpha_ij * j_j[j] * dF_j
                #print(tresholds_j)

                # Веса от входного слоя к скрытому
                # wki = wki - alpha * ji * dFi * yi
                for k in range(self.inputs):
                    for i in range(self.hiddens):
                        dFi = y_i[i] * (1 - y_i[i]) # derivative sigmoid func
                        self.weights_ki[k][i] -= self.alpha_ki * j_i[i] * dFi * y_i[i]
                #print(weights_ki)

                # Пороги скрытого слоя
                # Ti = Ti + alpha * ji * dFi
                for i in range(self.hiddens):
                    dFi = y_i[i] * (1 - y_i[i]) # derivative sigmoid func
                    self.tresholds_i += self.alpha_ki * j_i[i] * dFi
                #print(tresholds_i)

            # Вычисляем среднюю квадратичную ошибку
            E = 0
            for j in range(self.outputs):
                E += 1./2 * (y_j[j] - etalon[j]) ** 2

            # значение x для графика с ошибкой
            self.valuesXforGraph.append(E)

            eras += 1

            # значение y для графика с ошибкой
            self.valuesYforGraph.append(eras)

            # выводим значения в консоль
            #print('eras: %8d\tE: %32.20f\r' % (eras, E), end = '')
            print('eras: %8d\tE: %32.20f' % (eras, E))

            # условия останова
            if E < self.Ee or eras >= self.maxIterations:
                print()
                break

        print('Eras: %d with error E = %f' % (eras, E))

        print('\nFunction lab5 :: learning are end\n')

    def save(self):
        print('\nFunction lab5 :: save are start\n')

        fp = open('weights_ki.csv', 'w')
        for k in range(self.inputs):
            for i in range(self.hiddens):
                fp.write( str(self.weights_ki[k][i]) )
                if i != (self.hiddens - 1):
                    fp.write('\t')
            if k != (self.inputs - 1):
                fp.write('\n')
        fp.close()

        fp = open('weights_ij.csv', 'w')
        for i in range(self.hiddens):
            for j in range(self.outputs):
                fp.write( str(self.weights_ij[i][j]) )
                if j != (self.outputs - 1):
                    fp.write('\t')
            if i != (self.hiddens - 1):
                fp.write('\n')
        fp.close()

        fp = open('tresholds_i.csv', 'w')
        for i in range(self.hiddens):
            fp.write( str(self.tresholds_i[i]) )
            if i != (self.hiddens - 1):
                fp.write('\t')
        fp.close()

        fp = open('tresholds_j.csv', 'w')
        for j in range(self.outputs):
            fp.write( str(self.tresholds_j[j]) )
            if j != (self.outputs - 1):
                fp.write('\t')
        fp.close()

        print('\nFunction lab5 :: save are end\n')

    def test(self):
        print('\nFunction lab5 :: test are start\n')

        print('Save weights and tresholds:')
        print('y - yes')
        print('n - no')
        key = input()
        if key == 'y':
            self.save()

        print('\nFunction lab5 :: test are end\n')