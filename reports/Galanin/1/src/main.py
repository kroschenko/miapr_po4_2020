import math

def print_headTable():
	print("| %16s | %16s | %16s | %16s |" % (
		"y[]",
		"etalonoe zn",
		"Polychenoe zn",
		"Otklonenie"
	))
	print("| %16s | %16s | %16s | %16s |" % (
		"----------------",
		"----------------",
		"----------------",
		"----------------"
	))

a = 1
b = 9
d = 0.5
L = 4

alpha = 10e-6
Em = 10e-6

w = []
for i in range(L):
	w.append(0)

T = 1

m = 30
m2 = 15
etalon = []
for i in range(m + m2):
	etalon.append(0)

for i in range(m + m2):
	step = 0.1
	x = step * i
	etalon[i] = alpha * math.sin(b * x) + d

while 1:
	E = 0

	for i in range (m - L):
		y1 = 0
		for j in range(L):
			y1 += w[j] * etalon[i + j]
		y1 -= T

		for j in range(L):
			w[j] -= alpha * ( y1 - etalon[i + L] ) * etalon[i + j]

		T += alpha * (y1 - etalon[i + L])

		E += 0.5 * math.pow( (y1 - etalon[i + L]), 2)

	if E < Em:
		break;

print("Training sample:")
print_headTable()

trainingSample = []

for i in range(m + m2):
	trainingSample.append(0)

for i in range(m):
	trainingSample[i] = 0
	for j in range(L):
		trainingSample[i] += w[j] * etalon[j + i - L]

	trainingSample[i] -= T

	print("| %16d | %16lf | %16lf | %16lf |" % (
		i,
		etalon[i],
		trainingSample[i],
		etalon[i] - trainingSample[i]
	))

print("Forecasting the future:")
print_headTable()

for i in range(m2):
	trainingSample[i + m] = 0

	for j in range(L):
		trainingSample[i + m] += w[j] * etalon[m - L + j + i]

	trainingSample[i] += T

	print("| %16d | %16lf | %16lf | %16lf |" % (
		i + m,
		etalon[i],
		trainingSample[i],
		etalon[i] - trainingSample[i]
	))
