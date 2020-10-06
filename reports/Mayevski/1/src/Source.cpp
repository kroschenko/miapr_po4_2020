
#include <iostream>
#include <iomanip>
using namespace std;

int main() {
	setlocale(LC_ALL, "rus");
	system("color f0");
	int a = 1,
		b = 9,
		en = 4, //количество входов ИНС
		n = 30, //количесвто значений, на которых производится обучение
		predict = 15; //количесвто значений, на которых производится прогнозирование

	double d = 0.5,
		Em = 0.01, //минимальная среднеквадратичная ошибка сети
		E, //суммарная среднеквадратичная ошибка
		T = 1; //порог нейронной сети

	double* W = new double[en]; //весовые коэффициенты
	for (int i = 0; i < en; i++) { //задаем случайным образом весовые коэффициенты
		W[i] = static_cast <double> (rand()) / (static_cast <double>(RAND_MAX / 2));
	}

	double* etalon = new double[n + predict]; //эталонные значения y
	for (int i = 0; i < n + predict; i++) { //вычисляем эталонные значения
		double step = 0.1, //шаг
			x = step * i;
		etalon[i] = a * sin(b * x) + d;
	}

	do {
		double y1, //выходное значение нейронной сети
			A = 0.005; //скорость обучения
		E = 0;

		for (int i = 0; i < n - en; i++) {
			y1 = 0;

			for (int j = 0; j < en; j++) { //векторы выходной активности сети
				y1 += W[j] * etalon[i + j];
			}
			y1 -= T;

			for (int j = 0; j < en; j++) { //изменение весовых коэффициентов
				W[j] -= A * (y1 - etalon[i + en]) * etalon[i + j];
			}

			T += A * (y1 - etalon[i + en]); //изменение порога нейронной сети
			E += 0.5 * pow(y1 - etalon[i + en], 2); //расчет суммарной среднеквадратичной ошибки

		}
	} while (E > Em);

	cout << "РЕЗУЛЬТАТЫ ОБУЧЕНИЯ" << endl;
	cout << setw(27) << left << "Эталонные значения" << setw(23) << left << "Полученные значения" << "Отклонение" << endl;
	double* predicated_values = new double[n + predict];

	for (int i = 0; i < n; i++) {
		predicated_values[i] = 0;
		for (int j = 0; j < en; j++) {
			predicated_values[i] += W[j] * etalon[j + i ]; //получаемые значения в результате обучения
		}
		predicated_values[i] -= T;

		cout << "y[" << i << "] = " << setw(20) << left << etalon[i] << setw(23) << left;
		cout << predicated_values[i] << etalon[i] - predicated_values[i] << endl;
	}

	cout << endl << "РЕЗУЛЬТАТЫ ПРОГНОЗИРОВАНИЯ" << endl;
	cout << setw(28) << left << "Эталонные значения" << setw(23) << left << "Полученные значения" << "Отклонение" << endl;

	for (int i = 0; i < predict; i++) {
		predicated_values[i + n] = 0;
		for (int j = 0; j < en; j++) {
			//прогнозируемые значения
			predicated_values[i + n] += W[j] * etalon[n - en + j + i];
		}
		predicated_values[i + n] += T;

		cout << "y[" << n + i << "] = " << setw(20) << left << etalon[i + n] << setw(23) << left;
		cout << predicated_values[i + n] << etalon[i + n] - predicated_values[i + n] << endl;
	}

	delete[]etalon;
	delete[]predicated_values;
	delete[]W;
	return 0;
}
