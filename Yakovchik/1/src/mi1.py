import math
import random

def function(x, a, b, d):
	return a * math.sin(b * x) + d

a = 1
b = 5
d = 0.1

a_inputs = 3 				# Количество входов нс
a_testing = 30       # Кол-во элементов для обучения нс
a_learning = 15    # Количество элементов для тестирования нс
min_error = 0.0001                   # Минимальная среднеквадратичная ошибка
step = 0.1                        
s_training = 0.2                # Скорость обучения              

T = random.uniform(0.5, 1)          # Порог нс
synaptic_weights = []               # Синаптические веса


#рандомно генерируем веса
for i in range(a_inputs):
	synaptic_weights.append(random.uniform(0, 1))


training_outputs = []               # Эталонные выходные значения

for i in range(a_testing + a_learning):
	x = i * step
	training_outputs.append(function(x, a, b, d))

error = 1

print(T)
print(synaptic_weights)

while(error > min_error):
	error = 0						#Суммарная среднеквадратичная ошибка

	for i in range(a_testing):
		output = 0

		# Вычисляем выходное значение нейронной сети
		for j in range(a_inputs):
			output += (synaptic_weights[j] * training_outputs[j + i])

		output -= T

		#Корректируем порог нейронной сети, веса и ошибку
		for j in range(a_inputs):
			synaptic_weights[j] -= s_training * (output - training_outputs[i + a_inputs]) * training_outputs[i + j]
		
		T += s_training * (output - training_outputs[i + a_inputs]) 
		error += 0.5 * ((output - training_outputs[i + a_inputs]) ** 2)


outputs = []

print("Результаты обучения:")

for i in range(a_testing):
	outputs.append(0)

	for j in range(a_inputs):
		outputs[i] += synaptic_weights[j] * training_outputs[j + i - a_inputs]

	outputs[i] -= T

	print(str(i) + "   " + str(outputs[i]) + "   " + str(training_outputs[i]) + "   " + str(outputs[i] - training_outputs[i]))

print("Результаты прогнозирования:")

for i in range(a_learning):
	outputs.append(0)

	for j in range(a_inputs):
		outputs[i + a_testing] += synaptic_weights[j] * training_outputs[a_testing - a_inputs + j + i]

	outputs[i + a_testing] -= T

	print(str(i + a_testing) + "   " + str(outputs[i + a_testing]) + "   " + str(training_outputs[i + a_testing]) + "   " + str(outputs[i + a_testing] - training_outputs[i + a_testing]))
