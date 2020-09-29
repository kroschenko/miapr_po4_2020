import math
import random

a = 2
b = 5
d = 0.6
input = 5

alpha = 0.1
Em = 0.0001

T = 0.6
w = []

print("Весовые коэффициенты:")

for i in range(input):
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
    for i in range(n - input):
        y1 = 0
        for j in range(input):
            y1 += w[j] * et[i + j]
        y1 -= T

        for j in range(input):
            w[j] -= alpha * (y1 - et[i + input]) * et[i + j]
        T += alpha * (y1 - et[i + input])
        E += 0.5 * ((y1 - et[i + input]) ** 2)

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
for i in range(n):
    training.append(0)

    for j in range(input):
        training[i] += w[j] * et[j + i - input]
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

for i in range(n2):
    training.append(0)

    for j in range(input):
        training[i + n] += w[j] * et[n - input + j + i]

    training[i + n] -= T

    print(" %2d  %9lf  %18lf  %19lf " % (
        i + n,
        et[i + n],
        training[i + n],
        et[i + n] - training[i + n]
    ))