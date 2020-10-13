//вариант 9
#include <iostream>
#include <Windows.h>
#include <iomanip>
#include <ctime>
#include <cmath>

using namespace std;

float function(int a, int b, float x, float d);
void print_result(int n, float T, float* Y, int n_learn, int N, int n_protected, float* W);

int main() {
	SetConsoleCP(1251);
	SetConsoleOutputCP(1251);
	system("color f0");
	srand(time(0));
	int a = 1, b = 8, n = 5, n_learn = 30, n_predicted = 15;
	float d = 0.3, step = 0.1, x = 0, E, Ee = 0.01, T = 1;
	float* W = new float[n];
	for (int i = 0; i < n; i++) {
		W[i] = rand() % 100 * 0.1;
	}
	int N = n_learn + n_predicted;
	float* Y = new float[n_learn + n_predicted];
	for (int i = 0; i < N; i++) {
		x = x + step;
		Y[i] = function(a, b, x, d);
	}
	do {
		double y1, //выходное значение нейронной сети
			alpha = 0.3; //скорость обучения
		E = 0;
		for (int i = 0; i < n_learn - n; i++) {
			y1 = 0;

			for (int j = 0; j < n; j++) { //векторы выходной активности сети
				y1 += W[j] * Y[i + j];
			}
			y1 -= T;

			for (int j = 0; j < n; j++) { //изменение весовых коэффициентов
				W[j] -= alpha * (y1 - Y[i + n]) * Y[i + j];
			}

			T += alpha * (y1 - Y[i + n]); //изменение порога нейронной сети
			E += 0.5 * pow(y1 - Y[i + n], 2); //расчет суммарной среднеквадратичной ошибки
		}
	} while (E > Ee);
	print_result(n, T, Y, n_learn, N, n_predicted, W);
	delete[] Y;
	delete[] W;
	system("pause");
	return 0;
}

float function(int a, int b, float x, float d) {
	return a * sin(b * x) + d;
}

void print_result(int n, float T, float* Y, int n_learn, int N, int n_predicted, float* W) {
	cout << "РЕЗУЛЬТАТЫ:" << endl;
	cout << "1) Обучение:" << endl;
	cout << setw(30) << left << "Эталонные значения" << setw(30) << left << "Полученные значения" << "Отклонение" << endl;
	float* prediction = new float[N];
	for (int i = 0; i < n_learn; i++) {
		prediction[i] = 0;
		for (int j = 0; j < n; j++) {
			prediction[i] += W[j] * Y[j + i];
		}
		prediction[i] -= T;
		cout << "y[" << i << "] = " << setw(30) << left << Y[i+n] << setw(30) << left << prediction[i] << Y[i+n] - prediction[i] << endl;
	}
	cout << "2) Прогнозирование:" << endl;
	cout << setw(30) << left << "Эталонные значения" << setw(30) << left << "Полученные значения" << "Отклонение" << endl;
	for (int i = 0; i < n_predicted; i++) {
		prediction[i + n_learn] = 0;
		for (int j = 0; j < n; j++) {
			prediction[i + n_learn] += W[j] * Y[i + j + n_learn - n];
		}
		prediction[i + n_learn] -= T;
		cout << "y[" << i + n_learn << "] = " << setw(30) << left << Y[i + n_learn] << setw(30) << left << prediction[i+n_learn] << Y[i + n_learn] - prediction[i+n_learn] << endl;
	}
	delete[] prediction;
}