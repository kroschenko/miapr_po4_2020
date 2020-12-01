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
	return 0.3 * cos(0.5 * x) + 0.05 * sin(0.5 * x);
}
double* hidden(double x, double w1[3][8], double T[3])
{
	double* result = new double[3];
	for (int i = 0; i < 3; i++)
		result[i] = 0;
	double Inputs[8];
	for (int k = 0; k < 8; k++, x += 0.1)
		Inputs[k] = function(x);
	for (int i = 0; i < 3; i++)
	{
		for (int k = 0; k < 8; k++)
			result[i] += Inputs[k] * w1[i][k];
		result[i] -= T[i];
		result[i] = sigmoid(result[i]);
	}
	return result;
}
double output(double x, double w1[3][8], double w2[3], double T[3 + 1])
{
	double Result = 0;
	double* hidden_result = hidden(x, w1, T);
	for (int j = 0; j < 3; j++) {
		Result += hidden_result[j] * w2[j];
	}
	Result -= T[3];
	return Result;
}
int main()
{
	setlocale(LC_ALL, "rus");
	int epox = 0;
	double w1[3][8], w2[3], T[3 + 1], Reference, E_min = 0.00002, alpha = 0.4, x = 4, current, E = 0;
	for (int i = 0; i < 3; i++)
	{
		for (int k = 0; k < 8; k++)
		{
			w1[i][k] = ((double)rand() / RAND_MAX) * 0.005;
		}
		w2[i] = ((double)rand() / RAND_MAX) * 0.005;
		T[i] = ((double)rand() / RAND_MAX) * 0.005;
	}
	T[3] = ((double)rand() / RAND_MAX) * 0.005;
	do
	{
		E = 0;
		for (int q = 0; q < 200; q++)
		{
			current = output(x, w1, w2, T);
			Reference = function(x + 8 * 0.1);
			double error = current - Reference;
			double* Hiddens = hidden(x, w1, T);
			for (int j = 0; j < 3; j++)
				w2[j] -= alpha * error * Hiddens[j];
			T[3] += alpha * error;
			for (int k = 0; k < 3; k++)
			{
				for (int i = 0; i < 8; i++)
					w1[k][i] -= alpha * function(x + i * 0.1) * Hiddens[k] * (1 - Hiddens[k]) * w2[k] * error;
				T[k] += alpha * Hiddens[k] * (1 - Hiddens[k]) * w2[k] * error;
			}
			x += 0.1;
			E += pow(error, 2);
		}
		E /= 2;
		epox++;
	} while (E > E_min);
	cout << endl << epox << endl;
	cout << "Эталоное значение:" << setw(42) <<  "Полученное значение:" << setw(28) << "Отклонение: " <<  endl;
	for (int i = 0; i < 100; i++)
	{
		double Result = output(x, w1, w2, T), Ethalonn = function(x + 8 * 0.1);
		cout << fixed << setprecision(5) << Ethalonn << setw(21) << Result << setw(29) << Result - Ethalonn << endl;
		x += 0.1;
	}
	system("pause");
}