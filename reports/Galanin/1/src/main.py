import math
import random

def print_headTable():
	print("| %20s | %20s | %20s | %20s |" % (
		"Итерация",
		"Эталонное значение",
		"Полученное значение",
		"Отклонение"
	))
	print("| %16s | %16s | %16s | %16s |" % (
		"--------------------",
		"--------------------",
		"--------------------",
		"--------------------"
	))

a = 1
b = 9
d = 0.5
L = 4

alpha = 0.5
Em = 1e-6

w = []
for i in range(L):
	w.append(random.random() * 0.02 - 0.01)
	print("w[%d] = %lf" % (i, w[i]))

T = 1

m = 30
m2 = 15
e = []
for i in range(m + m2):
	step = 0.1
	x = step * i
	e.append(a * math.sin(b * x) + d)

while 1:
	E = 0
	for i in range (m - L):
		y1 = 0
		for j in range(L):
			y1 += w[j] * e[i + j]
		y1 -= T

		for j in range(L):
			w[j] -= alpha * ( y1 - e[i + L] ) * e[i + j]

		T += alpha * (y1 - e[i + L])

		E += 0.5 * math.pow( (y1 - e[i + L]), 2)

	if E < Em:
		break

print("Результаты обучение:")
print_headTable()

trainingSample = []

for i in range(m):
	trainingSample.append(0)

	if i % L == 0:
		print("%d эпоха" % (i / 4 + 1))
	
	for j in range(L):
		trainingSample[i] += w[j] * e[j + i - L]

	trainingSample[i] -= T

	print("| %20d | %20lf | %20lf | %20lf |" % (
		i,
		e[i],
		trainingSample[i],
		e[i] - trainingSample[i]
	))

print("Результаты прогнозирование:")
print_headTable()

for i in range(m2):
	trainingSample.append(0)

	if i % L == 0:
		print("%d эпоха" % (i / 4 + 1))

	for j in range(L):
		trainingSample[i + m] += w[j] * e[m - L + j + i]

	trainingSample[i] += T

	print("| %20d | %20lf | %20lf | %20lf |" % (
		i + m,
		e[i + m],
		trainingSample[i + m],
		e[i + m] - trainingSample[i + m]
	))
