import math
import random
import matplotlib.pyplot as plt

def print_headTable():
	print("| %20s | %20s | %20s | %20s |" % (
		"y[]",
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

T = 0.5

m = 30
m2 = 15

print("a = %d" % a)
print("b = %d" % b)
print("d = %f" % d)
print("L = %d" % L)
print("T = %f" % T)
print("alpha = %f" % a)
print("Em = %f" % Em)
print("m = %d" % m)
print("m2 = %d" % m2)

e = []
for i in range(m + m2):
	step = 0.1
	x = step * i
	e.append(a * math.sin(b * x) + d)

print("|%20s|%20s|" % ("Eras", "E"))
print("|%20s|%20s|" % (
	"--------------------",
	"--------------------"
))
eras = 0
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
		eras += 1

	plt.plot(eras, E, 'o-m') # точки на графике

	print("|%20d|%20f|" % (eras, E))

	if E < Em:
		break

print("\nEras %d\n" % eras)

print("Результаты обучение:")
print_headTable()

trainingSample = []

for i in range(m):
	trainingSample.append(0)
	
	for j in range(L):
		trainingSample[i] += w[j] * e[j + i]

	trainingSample[i] -= T

	print("| %20d | %20lf | %20lf | %20lf |" % (
		i,
		e[i + L],
		trainingSample[i],
		e[i + L] - trainingSample[i]
	))

print("Результаты прогнозирование:")
print_headTable()

for i in range(m2):
	trainingSample.append(0)

	for j in range(L):
		trainingSample[i + m] += w[j] * e[m - L + j + i]

	trainingSample[i + m] -= T

	print("| %20d | %20lf | %20lf | %20lf |" % (
		i + m,
		e[i + m],
		trainingSample[i + m],
		e[i + m] - trainingSample[i + m]
	))

plt.show()