import math
import random


a = 4
b = 7
d = 0.2
inputs = 4

alpha = 0.01
Em = 1e-6

T = 0.2 
w = []

print("Весовые коэффициенты:")

for i in range(inputs):
    w.append(random.random())
    print(w[i])

n = 30
n2 = 15
et = []
for i in range(n + n2):
    step = 0.1
    x = step * i
    et.append(a * math.sin(b * x) + d)

while 1:
    E = 0
    for i in range(n - inputs):
        y1 = 0
        for j in range(inputs):
            y1 += w[j] * et[i + j]
        y1 -= T
        for j in range(inputs):
            w[j] -= alpha * (y1 - et[i + inputs]) * et[i + j]
        T += alpha * (y1 - et[i + inputs])
        E += 0.5 * ((y1 - (et[i + inputs]) ** 2))
    if E < Em:
        break

print("Error: ", E)

print("\nРезультат тренировки:\n")
print(" %2s  %2s  %2s  %2s " % (
        "y[]",
        "Эталонное значение",
        "Полученное значение",
        "Отклонение"
    ))

training = []
for i in range(n):
    training.append(0)
    for j in range(inputs):
        training[i] += w[j] * et[j + i]
    training[i] -= T

    print(" %2d  %9lf  %18lf  %19lf " % (
        i,
        et[i + inputs],
        training[i],
        training[i] - et[i + inputs]
    ))

print("\nРезультат прогназированния:\n")
print(" %2s  %2s  %2s  %2s " % (
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

    print(" %2d  %9lf  %18lf  %19lf " % (
        i + n,
        et[i + n],
        training[i + n],
        training[i + n] - et[i + n]
    ))
