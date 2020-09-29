import math
import random

a = 2
b = 5
d = 0.6
L = 5

alpha = 0.1
Em = 0.0001

T = 0.6
w = []

print("Весовые коэффициенты:")

for i in range(L):
    w.append(random.random())
    print(w[i])

m = 30
m2 = 15
et = []
for i in range(m + m2):
    step = 0.1
    x = step * i
    et.append(a * math.sin(b * x) + d)

while 1:
    E = 0
    for i in range(m - L):
        y1 = 0
        for j in range(L):
            y1 += w[j] * et[i + j]
        y1 -= T

        for j in range(L):
            w[j] -= alpha * (y1 - et[i + L]) * et[i + j]
        T += alpha * (y1 - et[i + L])
        E += 0.5 * ((y1 - et[i + L]) ** 2)

    if E < Em:
        break


print("\nРезультат тренировки:\n")
print(" %2s  %2s  %2s  %2s " % (
        "y[]",
        "Эталонное значение",
        "Полученное значение",
        "Отклонение"
    ))

training = []
for i in range(m):
    training.append(0)

    for j in range(L):
        training[i] += w[j] * et[j + i - L]
    training[i] -= T

    print(" %2d  %9lf  %18lf  %19lf " % (
        i,
        et[i],
        training[i],
        et[i] - training[i]
    ))

print("\nРезультат прогназированния:\n")
print(" %2s  %2s  %2s  %2s " % (
        "y[]",
        "Эталонное значение",
        "Полученное значение",
        "Отклонение"
    ))

for i in range(m2):
    training.append(0)

    for j in range(L):
        training[i + m] += w[j] * et[m - L + j + i]

    training[i + m] -= T

    print(" %2d  %9lf  %18lf  %19lf " % (
        i + m,
        et[i + m],
        training[i + m],
        et[i + m] - training[i + m]
    ))