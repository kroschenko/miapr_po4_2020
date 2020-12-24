import math
import random

a = 1
b = 5
d = 0.1
inputs = 3

alpha = 0.1
Em = 1e-6

T = 0.6 
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

Era = 0
    
    
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
        E += 0.5 * ((y1 - et[i + inputs]) ** 2)
    
    Era += 1
    print("Era: %8d | Error: %20.18f" % (Era, E))
    if E < Em:
        print()
        break


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

print("\nРезультат прогнозированния:\n")
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
