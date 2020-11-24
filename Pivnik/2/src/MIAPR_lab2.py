import math
import random

def function(x, a, b, d):
    return a * math.sin(b * x) + d

a = 2
b = 5
d = 0.6

inputs = 5               
Em = 1e-8               
step = 0.1                          
alpha = 0.1                              

T = random.uniform(0.5, 1)       
w = []              

print("Весовые коэффициенты:")

for i in range(inputs):
    w.append(random.random())
    print(w[i])


et = []             
n = 30       
n2 = 15    
for i in range(n + n2):
    x = i * step
    et.append(function(x, a, b, d))

count = 0

while 1:

    E = 0

    for i in range(n):

        temp = 0
        for j in range(inputs):
           temp += (et[i + j])**2
        alpha = 1/(1 + temp)

        y = 0

        for j in range(inputs):
            y += (w[j] * et[j + i])

        y -= T

        for j in range(inputs):
            w[j] -= alpha * (y - et[i + inputs]) * et[i + j]
        T += alpha * (y - et[i + inputs]) 
        E += 0.5 * ((y - et[i + inputs]) ** 2)

    if E < Em:
        break
    
    print("Error: ", E)
    count += 1

print("Эпохи ", count)

training = []

print("Результаты обучения:")
print(" %2s %2s %2s %2s " % (
        "y[]",
        "Эталонное значение",
        "Полученное значение",
        "Отклонение"
    ))

for i in range(n):
    training.append(0)

    for j in range(inputs):
        training[i] += w[j] * et[j + i]

    training[i] -= T

    print(" %2d %9lf %18lf %19lf " % (
            i,
            et[i + inputs],
            training[i],
            training[i] - et[i + inputs]
        ))

print("Результаты прогнозирования:")
print(" %2s %2s %2s %2s " % (
        "y[]",
        "Эталонное значение",
        "Полученное значение",
        "Отклонение"
    ))

for i in range(n2):
    training.append(0)

    for j in range(inputs):
        training[i + n] += w[j] * et[n - inputs + j + i]

    training[i + n] -= T

    print(" %2d %9lf %18lf %19lf " % (
            i + n,
            training[i + n],
            et[i + n],
            training[i + n] - et[i + n]
        ))