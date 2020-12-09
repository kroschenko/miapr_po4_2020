#include <iostream>
#include <math.h>
#include <iomanip>
using namespace std;

double function(double x);
double sigmoid(double x);
double output(double x, double w1[3][8], double w2[3], double T[3 + 1]);
double* hidden(double x, double w1[3][8], double T[3]);


int main()
{
	double w1[3][8], w2[3], T[3 + 1], Reference, E_min = 0.0002, alpha = 0.4, x = 4, current, E = 0; int epoch = 0;
	for (int i = 0; i < 3; i++)
	{
		for (int k = 0; k < 8; k++)
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
		for (int q = 0; q < 200; q++) //обучающая выборка, при увеличении числа число epoch падает
		{
			current = output(x, w1, w2, T);
			Reference = function(x + 8 * 0.2);
			double error = current - Reference;
			double* Hiddens = hidden(x, w1, T);
			for (int j = 0; j < 3; j++)
				w2[j] -= alpha * error * Hiddens[j];
			T[4] += alpha * error;

			for (int k = 0; k < 3; k++)
			{
				for (int i = 0; i < 8; i++)
					w1[k][i] -= alpha * function(x + i * 0.2) * Hiddens[k] * (1 - Hiddens[k]) * w2[k] * error;
				T[k] += alpha * Hiddens[k] * (1 - Hiddens[k]) * w2[k] * error;
			}

			x += 0.2;
			E += pow(error, 2);

		}

		E /= 2;
		epoch++;

	} while (E > E_min);
	cout << "Eras: " << epoch << "\n";
	cout << "Ethalon value" << setw(15) << "Result " << setw(28) << "Delta " << endl;

	for (int i = 0; i < 100; i++)
	{
		double Resultat = output(x, w1, w2, T), Ethalonn = function(x + 8 * 0.2);
		cout << fixed << setprecision(5) << Ethalonn << setw(21) << Resultat << setw(29) << Resultat - Ethalonn << endl;
		x += 0.2;
	}

	system("pause");
}

double function(double x)
{
	return 0.3 * cos(0.5 * x) + 0.05 * sin(0.5 * x);
}

double sigmoid(double x)
{
	return 1 / (1 + pow(2, -x));
}

double output(double x, double w1[3][8], double w2[3], double T[3 + 1])
{
	double Resultat = 0;
	double* hidden_Result1 = hidden(x, w1, T);
	for (int j = 0; j < 3; j++) {
		Resultat += hidden_Result1[j] * w2[j];
	}
	Resultat -= T[4];
	return Resultat;
}

double* hidden(double x, double w1[3][8], double T[3])
{
	double* Result1 = new double[3];
	for (int i = 0; i < 3; i++)
		Result1[i] = 0;
	double Inputs[8];
	for (int k = 0; k < 8; k++, x += 0.2)
		Inputs[k] = function(x);
	for (int i = 0; i < 3; i++)
	{
		for (int k = 0; k < 8; k++)
			Result1[i] += Inputs[k] * w1[i][k];
		Result1[i] -= T[i];
		Result1[i] = sigmoid(Result1[i]);
	}
	return Result1;
}
