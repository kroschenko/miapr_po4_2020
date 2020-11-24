#include <iostream>
#include <Windows.h>
#include <iomanip>
#include <ctime>
#include <cmath>

using namespace std;

double function(int a, int b, double x, double d);
void print_result(int n, double T, double* Y, int n_learn, int N, int n_protected, double* W);

int main() {
	SetConsoleOutputCP(1251);
	system("color f0");
	srand(time(0));
	int a = 1, b = 5, n = 3, n_learn = 30, n_predicted = 15;
	double d = 0.1, step = 0.1, x1 = 0, E, Em = 0.001, T = 1;
	double* W = new double[n];
	for (int i = 0; i < n; i++) {
		W[i] = 1.0 / (double)rand();
	}
	int N = n_learn + n_predicted;
	double* Y = new double[N];
	for (int i = 0; i < N; i++) {
		x1 += step;
		Y[i] = function(a, b, x1, d);
	}
	double y1, //выходное значение нейронной сети
		alpha; //шаг обучения
	int epoh = 0;
	do {
		E = 0;
		for (int i = 0; i < n_learn - n; i++) {
			y1 = 0;

			double x2 = 0.0;
			for (int j = 0; j < n; j++) {
				x2 += pow(Y[i + j], 2);
			}
			alpha = 1 / (1 + x2); //адаптивный шаг

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
		epoh++;
	} while (E > Em);
	cout << epoh;
	print_result(n, T, Y, n_learn, N, n_predicted, W);
	delete[] Y;
	delete[] W;
	return 0;
}

double function(int a, int b, double x, double d) 
{
	return a * sin(b * x) + d;
}

void print_result(int n, double T, double* Y, int n_learn, int N, int n_predicted, double* W)
{
	cout << "РЕЗУЛЬТАТЫ:" << endl;
	cout << "1) Обучение:" << endl;
	cout << setw(30) << left << "Эталонные значения" << setw(38) << left << "Полученные значения" << "Отклонение" << endl;
	double* prediction = new double[N];
	for (int i = 0; i < n_learn; i++) {
		prediction[i] = 0;
		for (int j = 0; j < n; j++) {
			prediction[i] += W[j] * Y[j + i];
		}
		prediction[i] -= T;
		cout << "y[" << i << "] = " << setw(30) << left << Y[i + n] << setw(30) << left << prediction[i] << pow(Y[i + n] - prediction[i], 2) << endl;
	}
	cout << "2) Прогнозирование:" << endl;
	cout << setw(30) << left << "Эталонные значения" << setw(38) << left << "Полученные значения" << "Отклонение" << endl;
	for (int i = 0; i < n_predicted; i++) {
		prediction[i + n_learn] = 0;
		for (int j = 0; j < n; j++) {
			prediction[i + n_learn] += W[j] * Y[i + j + n_learn - n];
		}
		prediction[i + n_learn] -= T;
		cout << "y[" << i + n_learn << "] = " << setw(30) << left << Y[i + n_learn] << setw(30) << left << prediction[i + n_learn] << pow(Y[i + n_learn] - prediction[i + n_learn], 2) << endl;
	}
	delete[] prediction;
}