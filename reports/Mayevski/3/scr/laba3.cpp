#include <iostream>
#include <math.h>
#include <iomanip>
using namespace std;
double sigmoid(double x)
{
	return 1 / (1 + pow(2, -x));
}
double function(double x)
{
	return 0.2 * cos(0.6 * x) + 0.05 * sin(0.6 * x);
}
double* hidden(double x, double w1[4][10], double T[4])
{
	double* result = new double[4];
	for (int i = 0; i < 4; i++)
		result[i] = 0;
	double Inputs[10];
	for (int k = 0; k < 10; k++, x += 0.1)
		Inputs[k] = function(x);
	for (int i = 0; i < 4; i++)
	{
		for (int k = 0; k < 10; k++)
			result[i] += Inputs[k] * w1[i][k];
		result[i] -= T[i];
		result[i] = sigmoid(result[i]);
	}
	return result;
}
double output(double x, double w1[4][10], double w2[4], double T[4 + 1])
{
	double Result = 0;
	double* hidden_result = hidden(x, w1, T);
	for (int j = 0; j < 4; j++) {
		Result += hidden_result[j] * w2[j];
	}
	Result -= T[4];
	return Result;
}
int main()
{
	setlocale(LC_ALL, "rus");
	double w1[4][10], w2[4], T[4 + 1], Reference, E_min = 0.00002, alpha = 0.4, x = 4, current, E = 0;
	for (int i = 0; i < 4; i++)
	{
		for (int k = 0; k < 10; k++)
		{
			w1[i][k] = ((double)rand() / RAND_MAX) * 0.005;
		}
		w2[i] = ((double)rand() / RAND_MAX) * 0.005;
		T[i] = ((double)rand() / RAND_MAX) * 0.005;
	}
	T[4] = ((double)rand() / RAND_MAX) * 0.005;
	do
	{
		E = 0;
		for (int q = 0; q < 200; q++)
		{
			current = output(x, w1, w2, T);
			Reference = function(x + 10 * 0.1);
			double error = current - Reference;
			double* Hiddens = hidden(x, w1, T);
			for (int j = 0; j < 4; j++)
				w2[j] -= alpha * error * Hiddens[j];
			T[4] += alpha * error;
			for (int k = 0; k < 4; k++)
			{
				for (int i = 0; i < 10; i++)
					w1[k][i] -= alpha * function(x + i * 0.1) * Hiddens[k] * (1 - Hiddens[k]) * w2[k] * error;
				T[k] += alpha * Hiddens[k] * (1 - Hiddens[k]) * w2[k] * error;
			}
			x += 0.1;
			E += pow(error, 4);
		}
	} while (E > E_min);
	cout << "Эталоное значение " << setw(23) <<  "Полученное значение " << setw(20) << "Отклонение: " <<  endl;
	for (int i = 0; i < 100; i++)
	{
		double Result = output(x, w1, w2, T), Ethalonn = function(x + 10 * 0.1);
		cout << fixed << setprecision(5) << Ethalonn << setw(21) << Result << setw(29) << Result - Ethalonn << endl;
		x += 0.1;
	}
	system("pause");
}