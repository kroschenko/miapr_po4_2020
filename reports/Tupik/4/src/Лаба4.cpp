#include <iostream>
#include <math.h>
#include <iomanip>

using namespace std;

double act(double x); //функция активации (сигмоидная)
double func(double x); //функция
double* hidden(double x, double w1[4][10], double T[4]); //скрытый слой
double get_alpha(double w2[], double Error, double Output, double* Hiddens); //вычисление адаптивного шага обучения для сигмоидной функции активации
double output(double x, double w1[4][10], double w2[4], double T[5]); //выходное значение

int main()
{
	srand(time(0));
	setlocale(LC_ALL, "rus");
	int epoch = 0;
	double W1[4][10], W2[4], T[5], reference, E_min = 0.0002, alpha_new = 0.1, alpha = 0.1, x = 0, current, E = 0;
	for (int i = 0; i < 4; i++){
		for (int k = 0; k < 10; k++){
			W1[i][k] = (1 + rand() % 100) * 0.001;
		}
		W2[i] = (1 + rand() % 100) * 0.001;
		T[i] = (1 + rand() % 100) * 0.001;
	}
	T[4] = (1 + rand() % 100) * 0.001;
	do{
		E = 0;
		for (int q = 0; q < 500; q++){
			current = output(x, W1, W2, T);
			reference = func(x);
			double error = current - reference;
			double* hiddens = hidden(x, W1, T);
			for (int j = 0; j < 4; j++)
				W2[j] -= alpha * error * hiddens[j];
			T[4] += alpha * error;
			for (int k = 0; k < 4; k++){
				for (int i = 0; i < 10; i++)
					W1[k][i] -= alpha_new * func(x + i * 0.1) * hiddens[k] * (1 - hiddens[k]) * W2[k] * error;
				T[k] += alpha_new * hiddens[k] * (1 - hiddens[k]) * W2[k] * error;
			}
			alpha_new = get_alpha(W2, error, current, hiddens);
			x = x + 0.1;
			E += pow(error, 2);
		}
		E /= 2;
		epoch++;
		system("cls");
		cout << "Ошибка на эпохе " << epoch << " равна " << E;
	} while (E > E_min);
	cout << "\nКоличество эпох равно " << epoch << endl;
	cout << "Эталон" << setw(23) << "Результат" << setw(20) << "Ошибка " << endl;
	for (int i = 0; i < 45; i++){
		double res = output(x, W1, W2, T), etal = func(x);
		cout << fixed << setprecision(5) << etal << setw(21) << res << setw(29) << res - etal << endl;
		x = x + 0.1;
	}
	system("pause");
}

double act(double x){
	return 1 / (1 + pow(2.7, -x)); //сигмоидная функция
}

double func(double x){
	return 0.2 * cos(0.6 * x) + 0.05 * sin(0.6 * x);
}

double* hidden(double x, double w1[4][10], double T[4]){
	double* result = new double[4];
	for (int i = 0; i < 4; i++)
		result[i] = 0;
	double Inputs[10];
	for (int k = 0; k < 10; k++, x += 0.1)
		Inputs[k] = func(x);
	for (int i = 0; i < 4; i++)
	{
		for (int k = 0; k < 10; k++)
			result[i] += Inputs[k] * w1[i][k];
		result[i] -= T[i];
		result[i] = act(result[i]);
	}
	return result;
}

double get_alpha(double w2[], double Error, double Output, double* Hiddens){
	double alpha = 0, A = 0, B = 0;
	for (int i = 0; i < 4; i++)
	{
		A += pow(Error * w2[i] * (1 - Hiddens[i]) * Hiddens[i], 4) * Hiddens[i] * (1 - Hiddens[i]);
		B += pow(Error * w2[i] * (1 - Hiddens[i]) * Hiddens[i], 4) * Hiddens[i] * Hiddens[i] * (1 - Hiddens[i]) * (1 - Hiddens[i]);
	}
	alpha = 4 * A / (B * (1 + Output * Output));
	return alpha;
}

double output(double x, double w1[4][10], double w2[4], double T[5]){
	double Result = 0;
	double* hidden_result = hidden(x, w1, T);
	for (int j = 0; j < 4; j++) {
		Result += hidden_result[j] * w2[j];
	}
	Result -= T[4];
	return Result;
}