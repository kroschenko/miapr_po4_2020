#include <iostream>
#include <math.h>
#include <iomanip>

using namespace std;

double act(double x); //функция активации (сигмоидная)
double func(double x); //функция
double* hidden(double x, double w1[4][10], double T[4]); //скрытый слой
double output(double x, double w1[4][10], double w2[4], double T[5]); //выходное значение

int main(){
	srand(time(0));
	setlocale(LC_ALL, "rus");
	int epoch = 0;
	double W1[4][10], W2[4], T[5], reference, E_min = 0.0002, alpha = 0.1, x = 0, current, E = 0;
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
		for (int q = 0; q < 200; q++){
			current = output(x, W1, W2, T);
			reference = func(x);
			double error = current - reference;
			double* hiddens = hidden(x, W1, T);
			for (int j = 0; j < 4; j++)
				W2[j] -= alpha * error * hiddens[j];
			T[4] += alpha * error;
			for (int k = 0; k < 4; k++){
				for (int i = 0; i < 10; i++)
					W1[k][i] -= alpha * func(x + i * 0.1) * hiddens[k] * (1 - hiddens[k]) * W2[k] * error;
				T[k] += alpha * hiddens[k] * (1 - hiddens[k]) * W2[k] * error;
			}
			x = x + 0.1;
			E += pow(error, 2);
		}
		E /= 2;
		epoch++;
		system("cls");
		cout << "Ошибка на эпохе "<< epoch <<" равна " << E;
	} while (E > E_min);
	cout << "\nКоличество эпох равно "<< epoch;
	cout << "\nЭталон" << setw(23) << "Результат" << setw(29) << "Ошибка" << endl;
	for (int i = 0; i < 45; i++){
		double res = output(x, W1, W2, T), etal = func(x);
		cout << fixed << setprecision(5) << etal << setw(19) << res << setw(29) << res - etal << endl;
		x = x + 0.01;
	}
	system("pause");
}

double act(double x){
	return 1 / (1 + pow(2.7, -x)); //сигмоидная функция
}

double func(double x){
	return 0.6 * cos(0.3 * x) + 0.08 * sin(0.3 * x);
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

double output(double x, double w1[4][10], double w2[4], double T[5]){
	double Result = 0;
	double* hidden_result = hidden(x, w1, T);
	for (int j = 0; j < 4; j++) {
		Result += hidden_result[j] * w2[j];
	}
	Result -= T[4];
	return Result;
}