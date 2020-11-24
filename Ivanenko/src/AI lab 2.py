import math
import random

def function(x, a, b, d):
	return a * math.sin(b * x) + d

a = 2
b = 9
d = 0.4

amount_of_inputs = 3 				# Количество входов нейронной сети
amout_of_training_values = 30       # Количество элементов, на которых происходит обучение нейронной сети
amount_of_predicated_values = 15    # Количество элементов, на которых происходит тестирование нейронной сети
min_error = 0.001                 # Минимальная среднеквадратичная ошибка
step = 0.1                          # Шаг
training_speed = 0.1                # Скорость обучения              

T = random.uniform(0.5, 1)          # Порог нейронной сести

synaptic_weights = []               # Синаптические веса


#Случайно задаем синаптические веса нейронной сети
for i in range(amount_of_inputs):
	synaptic_weights.append(random.uniform(0, 1))


training_outputs = []               # Эталонные выходные значения

for i in range(amout_of_training_values + amount_of_predicated_values):
	x = i * step
	training_outputs.append(function(x, a, b, d))

error = 10

print(T)
print(synaptic_weights)

epochas = 0

while(error > min_error):

	error = 0						#Суммарная среднеквадратичная ошибка

	for i in range(amout_of_training_values):
		X = 0
		for j in range(amount_of_inputs):
			X += (training_outputs[i + j])**2
		training_speed = 1/(1 + X)

		output = 0

		# Вычисляем выходное значение нейронной сети
		for j in range(amount_of_inputs):
			output += (synaptic_weights[j] * training_outputs[j + i])

		output -= T

		#Корректируем порог нейронной сети, веса и ошибку
		for j in range(amount_of_inputs):
			synaptic_weights[j] -= training_speed * (output - training_outputs[i + amount_of_inputs]) * training_outputs[i + j]
		
		T += training_speed * (output - training_outputs[i + amount_of_inputs]) 
		error += 0.5 * ((output - training_outputs[i + amount_of_inputs]) ** 2)

	epochas+=1

print(str(epochas) + " -- ЭПОХИ")


outputs = []

print("Результаты обучения:")
print(" %2s %2s %2s %2s " % (
		"N",
		"Эталонное значение",
		"Полученное значение",
		"Ошибка"
	))

for i in range(amout_of_training_values):
	outputs.append(0)

	for j in range(amount_of_inputs):
		outputs[i] += synaptic_weights[j] * training_outputs[j + i]

	outputs[i] -= T

	print(" %2d %9lf %18lf %19lf " % (
			i,
			training_outputs[i + amount_of_inputs],
			outputs[i],
			outputs[i] - training_outputs[i + amount_of_inputs]
		))

print("Результаты прогнозирования:")
print(" %2s %2s %2s %2s " % (
		"N",
		"Эталонное значение",
		"Полученное значение",
		"Ошибка"
	))

for i in range(amount_of_predicated_values):
	outputs.append(0)

	for j in range(amount_of_inputs):
		outputs[i + amout_of_training_values] += synaptic_weights[j] * training_outputs[amout_of_training_values - amount_of_inputs + j + i]

	outputs[i + amout_of_training_values] -= T

	print(" %2d %9lf %18lf %19lf " % (
			i + amout_of_training_values,
			outputs[i + amout_of_training_values],
			training_outputs[i + amout_of_training_values],
			outputs[i + amout_of_training_values] - training_outputs[i + amout_of_training_values]
		))





