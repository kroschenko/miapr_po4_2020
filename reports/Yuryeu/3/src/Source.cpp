#include <iostream>
#include <math.h>
#include <iomanip>
#define input_layer 8
#define hidden_layer 3
using namespace std;
double sigmoid(double x)
{
	return 1 / (1 + pow(2, -x));
}

double function(double x)
{
	double a = 0.1, b = 0.1, c = 0.05, d = 0.1;
	return a * cos(b * x) + c * sin(d * x);
}

double* hidden(double x, double w1[hidden_layer][input_layer], double* T)
{
	double* result = new double[hidden_layer];
	for (int i = 0; i < hidden_layer; i++)
		result[i] = 0;
	double Inputs[input_layer];
	for (int k = 0; k < input_layer; k++, x += 0.1)
		Inputs[k] = function(x);
	for (int i = 0; i < hidden_layer; i++)
	{
		for (int k = 0; k < input_layer; k++)
			result[i] += Inputs[k] * w1[i][k];
		result[i] -= T[i];
		result[i] = sigmoid(result[i]);
	}
	return result;
}

double output(double x, double w1[hidden_layer][input_layer], double* w2, double* T)
{
	double Result = 0;
	double* hidden_result = hidden(x, w1, T);
	for (int j = 0; j < hidden_layer; j++) {
		Result += hidden_result[j] * w2[j];
	}
	Result -= T[4];
	return Result;
}

int main()
{
	setlocale(LC_ALL, "rus");
	int epox = 0;
	double w1[hidden_layer][input_layer], w2[hidden_layer], T[hidden_layer + 1], Reference, E_min = 0.00002, alpha = 0.4, x = 4, current, E = 0;
	for (int i = 0; i < hidden_layer; i++)
	{
		for (int k = 0; k < input_layer; k++)
		{
			w1[i][k] = ((double)rand() / RAND_MAX) * 0.05;
		}
		w2[i] = ((double)rand() / RAND_MAX) * 0.05;
		T[i] = ((double)rand() / RAND_MAX) * 0.05;
	}
	T[4] = ((double)rand() / RAND_MAX) * 0.05;
	do
	{
		E = 0;
		for (int q = 0; q < 1000; q++)
		{
			current = output(x, w1, w2, T);
			Reference = function(x + 8 * 0.1);
			double error = current - Reference;
			double* Hiddens = hidden(x, w1, T);
			for (int j = 0; j < hidden_layer; j++)
				w2[j] -= alpha * error * Hiddens[j];
			T[4] += alpha * error;
			for (int k = 0; k < hidden_layer; k++)
			{
				for (int i = 0; i < input_layer; i++)
					w1[k][i] -= alpha * function(x + i * 0.1) * Hiddens[k] * (1 - Hiddens[k]) * w2[k] * error;
				T[k] += alpha * Hiddens[k] * (1 - Hiddens[k]) * w2[k] * error;
			}
			x += 0.1;
			E += pow(error, 2);
		}
		E /= 2;
		epox++;

	} while (E > E_min);
	cout << epox << endl;
	cout << "Эталон" << setw(23) << "Прогноз" << setw(20) << "Отклонение" << endl;
	for (int i = 0; i < 100; i++)
	{
		double Result = output(x, w1, w2, T), Ethalonn = function(x + 8 * 0.1);
		cout << fixed << setprecision(5) << Ethalonn << setw(21) << Result << setw(29) << Result - Ethalonn << endl;
		x += 0.1;
	}
	system("pause");
}