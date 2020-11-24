#include <iostream>
#include <Windows.h>
#include <iomanip>
#include <ctime>
#include <cmath>

using namespace std;

double function(int a, int b, double x, double d);
void print_result(int n, double T, double* Y, int Num_Learning, int N, int n_protected, double* W);

int main() {
	SetConsoleOutputCP(1251);
	system("color f0");
	srand(time(0));
	int a = 1, b = 5, n = 3, Num_Learning = 30, Num_Predicted = 15;
	double d = 0.1, step = 0.1, x1 = 0, E, Em = 0.001, T = 1; 
	int epox = 0;
	double* W = new double[n];
	for (int i = 0; i < n; i++) {
		W[i] = 1.0 / (double)rand();
	}
	int N = Num_Learning + Num_Predicted;
	double* Y = new double[N];
	for (int i = 0; i < N; i++) {
		x1 += step;
		Y[i] = function(a, b, x1, d);
	}
	double y1, //выходное значение нейронной сети
		step_learning; //скорость обучения
	do {
		E = 0;
		for (int i = 0; i < Num_Learning - n; i++) {
			y1 = 0;

			double x2 = 0.0;
			for (int j = 0; j < n; j++) {
				x2 += pow(Y[i + j], 2);
			}
			step_learning = 1 / (1 + x2); //адаптивный шаг

			for (int j = 0; j < n; j++) { //векторы выходной активности сети
				y1 += W[j] * Y[i + j];
			}
			y1 -= T;

			for (int j = 0; j < n; j++) { //изменение весовых коэффициентов
				W[j] -= step_learning * (y1 - Y[i + n]) * Y[i + j];
			}

			T += step_learning * (y1 - Y[i + n]); //изменение порога нейронной сети
			E += 0.5 * pow(y1 - Y[i + n], 2); //расчет суммарной среднеквадратичной ошибки
		}
		epox++;
	} while (E > Em);
	cout << "Количество эпох: " << epox << endl;
	print_result(n, T, Y, Num_Learning, N, Num_Predicted, W);
	delete[] Y;
	delete[] W;
	return 0;
}

double function(int a, int b, double x, double d)
{
	return a * sin(b * x) + d;
}

void print_result(int n, double T, double* Y, int Num_Learning, int N, int Num_Predicted, double* W)
{
	cout << "РЕЗУЛЬТАТЫ:" << endl;
	cout << "1) Обучение:" << endl;
	cout << setw(30) << left << "Эталонные значения" << setw(38) << left << "Полученные значения" << "Отклонение" << endl;
	double* predict = new double[N];
	for (int i = 0; i < Num_Learning; i++) {
		predict[i] = 0;
		for (int j = 0; j < n; j++) {
			predict[i] += W[j] * Y[j + i];
		}
		predict[i] -= T;
		cout << "y[" << i << "] = " << setw(30) << left << Y[i + n] << setw(30) << left << predict[i] << pow(Y[i + n] - predict[i], 2) << endl;
	}
	cout << "2) Прогнозирование:" << endl;
	cout << setw(30) << left << "Эталонные значения" << setw(38) << left << "Полученные значения" << "Отклонение" << endl;
	for (int i = 0; i < Num_Predicted; i++) {
		predict[i + Num_Learning] = 0;
		for (int j = 0; j < n; j++) {
			predict[i + Num_Learning] += W[j] * Y[i + j + Num_Learning - n];
		}
		predict[i + Num_Learning] -= T;
		cout << "y[" << i + Num_Learning << "] = " << setw(30) << left << Y[i + Num_Learning] << setw(30) << left << predict[i + Num_Learning] << pow(Y[i + Num_Learning] - predict[i + Num_Learning], 2) << endl;
	}
	delete[] predict;
}