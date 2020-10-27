#include <iostream>
#include <iomanip>
#include <ctime>

using namespace std;

int main() {
	setlocale(0, "");
	int a = 3,
		b = 5,
		enter = 4, //кол-во входов ИНС
		N = 30, //количество значений для обучения
		val = 15; //количество значений для прогноза

	double d = 0.5,
		Em = 0.05, //минимальная среднеквадратичная ошибка сети
		E, //суммарная среднеквадратичная ошибка сети
		T = 1; //порог нейронной сети

	double* W = new double[enter]; //весовые коэффициенты (3)
	for (int i = 0; i < enter; i++) { //рандом весовых коэффициентов
		W[i] = (double)(rand()) / RAND_MAX;
		//cout << "W[" << i << "] = " << W[i] << endl; 
	}
	cout << endl;

	double* ref_values = new double[N + val]; //эталонные значения y
	for (int i = 0; i < N + val; i++) { //вычисляем эталонные значения
		double step = 0.1; //шаг
		double x = step * i;
		ref_values[i] = a * sin(b * x) + d;
	}
	int count = 0; //для индексов

	while (1) {
		double y1; //выходное значение нейронной сети
		double A = 0.04; //скорость обучения
		E = 0; //ошибка

		for (int i = 0; i < N - enter; i++) {
			y1 = 0;

			for (int j = 0; j < enter; j++) { //векторы выходной активности сети
				y1 += W[j] * ref_values[j + i];
			}
			y1 -= T;

			for (int j = 0; j < enter; j++) { //изменение весовых коэффициентов
				W[j] -= A * (y1 - ref_values[i + enter]) * ref_values[i + j];
			}

			T += A * (y1 - ref_values[i + enter]); //изменение порога нейронной сети
			E += 0.5 * pow(y1 - ref_values[i + enter], 2); //расчет суммарной среднеквадратичной ошибки
			count++;
			double temp = 0.0;
			for (int j = 0; j < enter; j++) {
				temp += pow(ref_values[i + j], 2);
			}
			A = 1 / (1 + temp); //адаптивный шаг
		}

		cout << count << " | " << E << endl;
		if (E < Em) break;
	}  //далее сеть обучена
	cout << endl;

	cout << "Результаты обучения" << endl;
	cout << setw(27) << right << "Эталонные значения" << setw(23) << right << "Полученные значения";
	cout << setw(23) << right << "Отклонение" << endl;
	double* predict_values = new double[N + val];

	for (int i = 0; i < N; i++) {
		predict_values[i] = 0;
		for (int j = 0; j < enter; j++) {
			predict_values[i] += W[j] * ref_values[j + i]; //получаемые значения в результате обучения
		}
		predict_values[i] -= T;

		cout << "y[" << i + 1 << "] = " << setw(20) << right << ref_values[i + enter] << setw(23) << right;
		cout << predict_values[i] << setw(23) << right << ref_values[i + enter] - predict_values[i] << endl;
	}

	cout << endl << "Результаты прогнозирования" << endl;
	cout << setw(28) << right << "Эталонные значения" << setw(23) << right << "Полученные значения" << setw(23) << right << "Отклонение" << endl;

	for (int i = 0; i < val; i++) {
		predict_values[i + N] = 0;
		for (int j = 0; j < enter; j++) {
			//прогнозируемые значения
			predict_values[i + N] += W[j] * ref_values[N - enter + j + i];
		}
		predict_values[i + N] -= T;

		cout << "y[" << N + i + 1 << "] = " << setw(20) << right << ref_values[i + N] << setw(23) << right;
		cout << predict_values[i + N] << setw(23) << right << ref_values[i + N] - predict_values[i + N] << endl;
	}

	delete[]ref_values;
	delete[]predict_values;
	delete[]W;

	system("pause");
	return 0;
}