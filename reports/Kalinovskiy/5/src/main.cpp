#include <iostream> 
using namespace std;
double Sigmoid(double x);
double* hidden_value(bool* entrances, double W12[20][40], double THid[]);
double* result_value(bool* entrances, double W12[20][40], double THid[], double W23[40][3], double TOut[], double hidden[40]);

int main()
{
	setlocale(LC_ALL, "rus");
	int eras = 0;
	bool Vect1[] = { 1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0 };
	bool Vect2[] = { 1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0 };
	bool Vect3[] = { 1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1 };
	bool* entrances = new bool[20];
	for (int i = 0; i < 20; i++) entrances[i] = 0;
	bool** Vectors = new bool* [9];
	Vectors[0] = Vect1;
	Vectors[1] = Vect2;
	Vectors[2] = Vect3;
	double W12[20][40], W23[40][3], THid[40], TOut[3], E_min = 0.001, Alpha = 0.04, Ethalon, E = 0, Outputs[3] = { 0 };
	double* Currents = new double[3];
	double* hidden = new double[40];
	double Error[3] = { 0 };
	double Ethalons[3] = { 0 };
	double ErrorHid[40] = { 0 };

	for (int i = 0; i < 20; i++)
	{
		for (int j = 0; j < 40; j++)
		{
			W12[i][j] = ((double)rand() / (RAND_MAX)) - 0.5;
			for (int k = 0; k < 3; k++)
			{
				W23[j][k] = ((double)rand() / (RAND_MAX)) - 0.5;
				TOut[k] = ((double)rand() / (RAND_MAX)) - 0.5;
			}
			THid[j] = ((double)rand() / (RAND_MAX)) - 0.5;
		}
	}
	do
	{
		E = 0;
		for (int N = 0; N < 3; N++)
		{
			Ethalons[0] = Ethalons[1] = Ethalons[2] = 0;
			Ethalons[N] = 1;
			entrances = Vectors[N];
			hidden = hidden_value(entrances, W12, THid);
			Currents = result_value(entrances, W12, THid, W23, TOut, hidden);
			for (int i = 0; i < 3; i++)
				Error[i] = Currents[i] - Ethalons[i];

			for (int j = 0; j < 40; j++)
			{
				for (int m = 0; m < 3; m++) {
					ErrorHid[j] += Error[m] * Currents[m] * (1 - Currents[m]) * W23[j][m];
				}
			}
			for (int j = 0; j < 3; j++)
			{
				for (int i = 0; i < 40; i++)
				{
					W23[i][j] -= Alpha * Error[j] * Currents[j] * (1 - Currents[j]) * hidden[i];
				}
				TOut[j] += Alpha * Error[j] * Currents[j] * (1 - Currents[j]);
			}

			for (int j = 0; j < 40; j++)
			{
				for (int i = 0; i < 20; i++)
				{
					W12[i][j] -= Alpha * ErrorHid[j] * hidden[j] * (1 - hidden[j]) * entrances[i];
				}
				THid[j] += Alpha * ErrorHid[j] * hidden[j] * (1 - hidden[j]);
			}
			E += pow(Error[N], 2);
		}
		E /= 2;
		cout << E << endl;
		eras++;
	} while (E > E_min);
	cout << eras << endl;


	double* HiddenPred;
	double* Values;
	bool Vectors3[] = { 1,1,0,0,1,1,0,1,1,1,0,0,1,1,0,0,1,0,0,0 };
	bool Vectors4[] = { 0,0,1,0,1,1,0,0,1,1,0,1,1,1,0,0,1,1,0,1 };
	bool Vectors5[] = { 0,0,1,1,1,1,0,0,1,0,1,0,1,1,0,0,1,0,0,0 };
	bool Vectors6[] = { 0,1,1,0,1,0,0,0,1,1,0,1,0,0,0,0,1,1,1,1 };
	bool Vectors7[] = { 1,1,1,0,1,1,0,0,0,1,1,0,0,0,1,0,1,1,1,0 };
	bool Vectors8[] = { 1,1,1,0,1,1,1,0,1,1,1,1,0,0,0,0,1,0,1,1 };
	bool Vectors9[] = { 0,1,1,0,0,1,1,1,1,0,1,0,1,0,1,0,0,0,1,1 };
	Vectors[3] = Vectors3;
	Vectors[4] = Vectors4;
	Vectors[5] = Vectors5;
	Vectors[6] = Vectors6;
	Vectors[7] = Vectors7;
	Vectors[8] = Vectors8;
	Vectors[9] = Vectors9;

	for (int i = 0; i < 10; i++)
	{
		entrances = Vectors[i];
		cout << "Вектор " << i + 1;
		for (int j = 0; j < 20; j++)
		{
			cout << entrances[j] << ' ';
		}
		cout << endl << "Результат: ";
		HiddenPred = hidden_value(entrances, W12, THid);
		Values = result_value(entrances, W12, THid, W23, TOut, HiddenPred);
		cout << Values[0] << ' ' << Values[1] << ' ' << Values[2] << endl;
		cout << endl;
	}
	system("pause");
}

double Sigmoid(double x)
{
	return 1 / (1 + pow(2.7, -x));
}

double* hidden_value(bool* entrances, double W12[20][40], double THid[])
{
	double* hidden = new double[40];
	for (int i = 0; i < 40; i++) hidden[i] = 0;
	for (int i = 0; i < 40; i++)
	{
		for (int j = 0; j < 20; j++)
		{
			hidden[i] += W12[j][i] * entrances[j];
		}
		hidden[i] -= THid[i];
		hidden[i] = Sigmoid(hidden[i]);
	}
	return hidden;
}

double* result_value(bool* entrances, double W12[20][40], double THid[], double W23[40][3], double TOut[], double hidden[40])
{
	double* Results = new double[3];
	for (int i = 0; i < 3; i++)
		Results[i] = 0;
	for (int j = 0; j < 3; j++)
	{
		for (int i = 0; i < 40; i++)
		{
			Results[j] += hidden[i] * W23[i][j];
		}
		Results[j] -= TOut[j];
		Results[j] = Sigmoid(Results[j]);
	}
	return Results;
}