#include <iostream>
#include <math.h>
#include <iomanip>

using namespace std;

double Sigmoid(double x) {
	return 1 / (1 + pow(2, -x));
}

double func(double x) {
	double a = 0.3, b = 0.3, c = 0.07, d = 0.3;
	return a * cos(b * x) + c * sin(d * x);
}

double* hidden(double x, double Wes1[4][10], double T[4]) {
	double* result_value = new double[4];
	for (int i = 0; i < 4; i++) {
		result_value[i] = 0;
	}
	double entrances[10];
	for (int k = 0; k < 10; k++, x += 0.1) {
		entrances[k] = func(x);
	}
	for (int i = 0; i < 4; i++) {
		for (int k = 0; k < 10; k++) {
			result_value[i] += entrances[k] * Wes1[i][k];
		}
		result_value[i] -= T[i];
		result_value[i] = Sigmoid(result_value[i]);
	}
	return result_value;
}
double output(double x, double Wes1[4][10], double Wes2[4], double T[4 + 1]) {
	double result = 0;
	double* hidden_result_value = hidden(x, Wes1, T);
	for (int j = 0; j < 4; j++) {
		result += hidden_result_value[j] * Wes2[j];
	}
	result -= T[4];
	return result;
}
int main() {
	setlocale(0, "");
	double Wes1[4][10], Wes2[4], T[4 + 1], reference_value, E_min = 0.00002, alpha = 0.4, x = 4, current, E = 0;
	for (int i = 0; i < 4; i++) {
		for (int k = 0; k < 10; k++) {
			Wes1[i][k] = ((double)rand() / RAND_MAX);
		}
		Wes2[i] = ((double)rand() / RAND_MAX);
		T[i] = ((double)rand() / RAND_MAX);
	}
	T[4] = ((double)rand() / RAND_MAX);
	do
	{
		E = 0;
		for (int q = 0; q < 200; q++)
		{
			current = output(x, Wes1, Wes2, T);
			reference_value = func(x + 10 * 0.1);
			double Error = current - reference_value;
			double* Hiddens = hidden(x, Wes1, T);
			for (int j = 0; j < 4; j++)
				Wes2[j] -= alpha * Error * Hiddens[j];
			T[3] += alpha * Error;
			for (int k = 0; k < 4; k++)
			{
				for (int i = 0; i < 10; i++)
					Wes1[k][i] -= alpha * func(x + i * 0.1) * Hiddens[k] * (1 - Hiddens[k]) * Wes2[k] * Error;
				T[k] += alpha * Hiddens[k] * (1 - Hiddens[k]) * Wes2[k] * Error;
			}
			x += 0.1;
			E += pow(Error, 4);
		}
		E /= 4;
		cout << "Error: " << E << endl;
	} while (E > E_min);
	cout << "N" << "\t" << "Эталон" << setw(20) << "Получ.знач." << setw(15) << "Отклонение" << endl;
	for (int i = 0; i < 15; i++) {
		double result = output(x, Wes1, Wes2, T), ethelon_value = func(x + 10 * 0.1);
		cout << i + 1 << "\t" << ethelon_value << right << setw(15) << result << right << setw(15) << result - ethelon_value << endl;
		x += 0.1;
	}
	system("pause");
	return 0;
}