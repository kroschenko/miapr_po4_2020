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
	return 0.4 * cos(0.2 * x) + 0.07 * sin(0.2 * x);
}
double* hidden(double x, double w1[3][8], double T[3])
{
	double* Resultatik = new double[3];
	for (int i = 0; i < 3; i++)
		Resultatik[i] = 0;
	double Inputs[8];
	for (int k = 0; k < 8; k++, x += 0.1)
		Inputs[k] = function(x);
	for (int i = 0; i < 3; i++)
	{
		for (int k = 0; k < 8; k++)
			Resultatik[i] += Inputs[k] * w1[i][k];
		Resultatik[i] -= T[i];
		Resultatik[i] = sigmoid(Resultatik[i]);
	}
	return Resultatik;
}
double output(double x, double w1[3][8], double w2[3], double T[3 + 1])
{
	double Resultat = 0;
	double* hidden_resultatik = hidden(x, w1, T);
	for (int j = 0; j < 3; j++) {
		Resultat += hidden_resultatik[j] * w2[j];
	}
	Resultat -= T[4];
	return Resultat;
}
int main()
{
	setlocale(LC_ALL, "rus");
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
	T[4] = ((double)rand() / RAND_MAX) * 0.005;
	do
	{
		E = 0;
		for (int q = 0; q < 200; q++)
		{
			current = output(x, w1, w2, T);
			Reference = function(x + 8 * 0.1);
			double oshibka = current - Reference;
			double* Hiddens = hidden(x, w1, T);
			for (int j = 0; j < 3; j++)
				w2[j] -= alpha * oshibka * Hiddens[j];
			T[4] += alpha * oshibka;
			for (int k = 0; k < 3; k++)
			{
				for (int i = 0; i < 8; i++)
					w1[k][i] -= alpha * function(x + i * 0.1) * Hiddens[k] * (1 - Hiddens[k]) * w2[k] * oshibka;
				T[k] += alpha * Hiddens[k] * (1 - Hiddens[k]) * w2[k] * oshibka;
			}
			x += 0.1;
			E += pow(oshibka, 3);
		}
		E /= 3;
		cout << "Oshibka: " << E << endl;
	} while (E > E_min);
	for (int i = 0; i < 100; i++)
	{
		double Resultat = output(x, w1, w2, T), Ethalonn = function(x + 8 * 0.1);
		cout << "Etalonnoe znachenie: " << fixed << setprecision(5) << Ethalonn << " Poluchennoe znachenie: " << Resultat << " Otklonenie: " << Resultat - Ethalonn << endl;
		x += 0.1;
	}
	system("pause");
}
