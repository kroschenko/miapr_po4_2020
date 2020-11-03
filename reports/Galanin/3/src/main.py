import random
import math
import numpy as np
import matplotlib.pyplot as plt
from etalons import get_etalons
from sigm import sigm

L = 8 # Количество входов ИНС
Lhid = 3 # Количество НЭ скрытого слоя

e = get_etalons(30) # эталонные значения

w = np.zeros((L, Lhid)) # массив размером [L x Lhid] для весов от входа к скрытому слою

for i in range(L):
    for j in range(Lhid):
        w[i][j] = random.uniform(0, 1)

ws = np.zeros(Lhid) # массив размером [Lhid x 1] для весов от скрытого слоя к выходу
for i in range(Lhid):
    w[i] = random.uniform(0, 1)

T = np.zeros(Lhid) # пороги от входов к срытому слою
for i in range(Lhid):
    T[i] = random.uniform(0, 1)

Ts = random.uniform(0, 1) # порог от скрытого слоя к выходу

Ee = 1e-6 # желаемая квадратичная ошибка, до значения которой мы хотим обучить сеть

alpha = 0.5 # 0 < alpha < 1 - скорость обучения

x = np.zeros(L)
for i in range(L):
    x[i] = e[i]
#print('x =\n%s\n' % x)
#print('w =\n%s\n' % w)

eras = 0
valueXforGraph = []
valueYforGraph = []
while 1:
    for k in range(30 - Lhid):
        S = x.dot(w) - T # S = [x] * [w] - [T] 
        #print('S = x * w - T =\n%s\n' % S)

        for i in range(Lhid):
            S[i] = sigm(S[i]) # [sigm(S1), sigm(S2), ..., sigm(SLhid)]
        y = S
        print('y = [sigm(S1), ..., sigm(SLhid)] =\n%s\n' % y)

        Y = y.dot(ws) - Ts # Y = [y] * [ws] - [Ts]
        #print('Y = [y] * [ws] - [Ts] =\n%s\n' % Y)

        gamma = Y - e[k]
        #print('gamma =\n%s\n' % gamma)

        ws = ws - alpha * gamma * 1 * y # [ws] = [ws] - a * j * 1 * Y
        #print('ws =\n%s\n' % ws)

        Ts = Ts + alpha * gamma * 1 # [Ts] = [Ts] + a * j * 1
        #print('Ts =\n%s\n' % Ts)

        gamma_s = np.zeros(Lhid)
        for i in range(Lhid):
            gamma_s[i] = y[i] - e[k + i]

        for i in range(L):
            for j in range(Lhid):
                w[i][j] -= alpha * gamma_s[j] * y[j] * (1 - y[j]) * y[j]
        #print('w = w - a * j * y * (1-y) * y =\n%s\n' % w)

        for i in range(Lhid):
            T[i] += alpha * gamma_s[i] * y[i] * (1 - y[i])
        #print('T = T + alpha * j * y * (1-y) =\n%s\n' % T)

        E = 0.5 * (Y - e[k]) ** 2
        #print('E = %s' % E)

    eras += 1

    valueXforGraph.append(eras)
    valueYforGraph.append(E)
    print('%24s %24s' % (eras, E))

    if E < Ee:
        break

plt.plot(valueXforGraph, valueYforGraph, 'Db', label="E")

plt.title("Error change graph") # Python write title in graph
plt.legend() # Python write legend in graph
plt.show() # Python open new windows and show graph