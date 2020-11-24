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
	return 0.4 * cos(0.4 * x) + 0.08 * sin(0.4 * x);
}
double* hidden(double x, double w1[2][6], double T[2])
{
	double* Resultatik = new double[2];
	for (int i = 0; i < 2; i++)
		Resultatik[i] = 0;
	double Inputs[6];
	for (int k = 0; k < 6; k++, x += 0.1)
		Inputs[k] = function(x);
	for (int i = 0; i < 2; i++)
	{
		for (int k = 0; k < 6; k++)
			Resultatik[i] += Inputs[k] * w1[i][k];
		Resultatik[i] -= T[i];
		Resultatik[i] = sigmoid(Resultatik[i]);
	}
	return Resultatik;
}
double get_alpha(double w2[], double Error, double Output, double Hiddens[])
{
	double alpha = 0, A = 0, B = 0;
	for (int i = 0; i < 2; i++)
	{
		A += pow(Error * w2[i] * (1 - Hiddens[i]) * Hiddens[i], 2) * Hiddens[i] * (1 - Hiddens[i]);
		B += pow(Error * w2[i] * (1 - Hiddens[i]) * Hiddens[i], 2) * Hiddens[i] * Hiddens[i] * (1 - Hiddens[i]) * (1 - Hiddens[i]);
	}
	alpha = 4 * A / (B * (1 + Output * Output));
	return alpha;
}
double output(double x, double w1[2][6], double w2[2], double T[2 + 1])
{
	double Resultat = 0;
	double* hidden_resultatik = hidden(x, w1, T);
	for (int j = 0; j < 2; j++) {
		Resultat += hidden_resultatik[j] * w2[j];
	}
	Resultat -= T[4];
	return Resultat;
}
int main()
{
	setlocale(LC_ALL, "rus");
	double w1[2][6], w2[2], T[2 + 1], Reference, E_min = 0.00002, alpha2 = 0.4, alpha = 0.4, x = 4, current, E = 0;
	for (int i = 0; i < 2; i++)
	{
		for (int k = 0; k < 6; k++)
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
			Reference = function(x + 6 * 0.1);
			double oshibka = current - Reference;
			double* Hiddens = hidden(x, w1, T);
			for (int j = 0; j < 2; j++)
				w2[j] -= alpha * oshibka * Hiddens[j];
			T[4] += alpha * oshibka;
			for (int k = 0; k < 2; k++)
			{
				for (int i = 0; i < 6; i++)
					w1[k][i] -= alpha2 * function(x + i * 0.1) * Hiddens[k] * (1 - Hiddens[k]) * w2[k] * oshibka;
				T[k] += alpha2 * Hiddens[k] * (1 - Hiddens[k]) * w2[k] * oshibka;
			}
			alpha2 = get_alpha(w2, oshibka, current, Hiddens);
			x += 0.1;
			E += pow(oshibka, 2);
		}
		E /= 2;
		cout << "Oshibka: " << E << endl;
	} while (E > E_min);
	for (int i = 0; i < 100; i++)
	{
		double Resultat = output(x, w1, w2, T), Etalonn = function(x + 6 * 0.1);
		cout << "Etalonnoe znachenie: " << fixed << setprecision(5) << Etalonn << " Poluchennoe znachenie: " << Resultat << " Otklonenie: " << Resultat - Etalonn << endl;
		x += 0.1;
	}
	system("pause");
}
