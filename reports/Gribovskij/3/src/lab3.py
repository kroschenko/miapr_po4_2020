import math
import random

def func(x):
    a = 0.2
    b = 0.6
    c = 0.05
    d = 0.6
    return a * math.cos(b * x) + c * math.sin(d * x)

def sigmoid(x):
    e = 2.718
    return 1 / (1 + e ** (-x))

def derivative(x):
    return x * (1 - x)

n_in = 10  #количество входов
n_el = 4   #количество элементов скрытого слоя
n_ob = 30  #размер выборки для обучения
n_pr = 15  #размер выборки для прогназированния
alpha = 0.1 #скорость обучения
Em = 1e-6  #минимальная ошибка

Err   = [0 for i in range(n_ob - n_in)]
E_hid = [] #ошибка скрытого слоя
F_hid = [] #функция активности скрытого слоя
Y_hid = [] #выходная активность скрытого слоя, выходного слоя
T_hid = [] #пороговые значение скрытого слоя, выходного слоя
W_hid = [[0] * n_el for i in range(n_in)] #весовые коэффициенты скрытого слоя
W_out = [] #весовые коэффициенты выходного слоя
Y_out = 0.0

t = []; #массив эталонных значений
y = [];  #массив обучаемых значений
z = [];  #массив прогнозируемых значений

for i in range(n_pr + n_ob + n_in):
    t.append(func(i * 0.1))

T_out = random.uniform(-0.1, 0.1)

print("Случайно заданные весовые коэффициенты: ")
for j in range(n_el):
    Y_hid.append(0)
    F_hid.append(0)
    for i in range(n_in):
        W_hid[i][j] = round(random.uniform(-0.1, 0.1), 2)
    W_out.append(round(random.uniform(-0.1, 0.1), 2))
    T_hid.append(round(random.uniform(-0.1, 0.1), 2))

print(W_hid)
print("\n")

count = 0
while True:
    count += 1
    E = 0.0
    for k in range(n_ob - n_in):
        Y_out = 0.0
        for i in range(n_el):
            for j in range(n_in):
                Y_hid[i] += W_hid[j][i] * t[k + j]
            Y_hid[i] -= T_hid[i]
            F_hid[i]  = sigmoid(Y_hid[i])
            Y_hid[i]  = 0.0
            Y_out    += F_hid[i] * W_out[i]
        Y_out -= T_out;
        Err[k] = Y_out - t[k];
        T_out += alpha * Err[k];
        for i in range(n_el):
            W_out[i] -= alpha * Err[k] * F_hid[i]
        for i in range(n_el):
            deriv = derivative(F_hid[i])
            for j in range(n_in):
                W_hid[j][i] -= alpha * Err[k] * deriv * W_out[i] * t[k + j]
            T_hid[i] += alpha * Err[k] * deriv * W_out[i]
        E += 0.5 * Err[k] ** 2
    #print(E," ",Em)
    if E < Em:
        break

print("Эпохи: ", count)
print("Ошибка", E)

#Обучение
print("Результаты обучения:")
print(" %2s %2s %2s %2s " % (
        "y[]",
        "Эталонное значение",
        "Полученное значение",
        "Отклонение"
    ))

for k in range(n_ob):
    y.append(0)
    for i in range(n_el):
        for j in range(n_in):
            y[k] += W_hid[j][i] * t[k + j]
        Y_hid[i] -= T_hid[i];
        F_hid[i]  = sigmoid(Y_hid[i]); #функция активации
        Y_hid[i]  = 0;
        y[k]     += F_hid[i] * W_out[i];
    y[k] -= T_out

    print(" %2d %9lf %18lf %19lf " % (
        k,
        t[k],
        y[k],
        t[k] - y[k]
    ))

#Прогнозированние
print("Результаты прогнозирования:")
print(" %2s %2s %2s %2s " % (
        "y[]",
        "Эталонное значение",
        "Полученное значение",
        "Отклонение"
    ))

for k in range(n_pr):
    z.append(0)
    for i in range(n_el):
        for j in range(n_in):
            Y_hid[i] += W_hid[j][i] * t[k + n_ob + j]
        Y_hid[i] -= T_hid[i];
        F_hid[i]  = sigmoid(Y_hid[i]); # функция активации
        Y_hid[i]  = 0;
        z[k]     += F_hid[i] * W_out[i];
    z[k] -= T_out

    print(" %2d %9lf %18lf %19lf " % (
            k + n_ob,
            t[k],
            z[k],
            t[k] - z[k]
        ))
