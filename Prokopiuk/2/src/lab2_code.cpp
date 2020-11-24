#include <iostream>
#include <iomanip>
#include <ctime>

using namespace std;

int main() {
	setlocale(0, "");
	int a = 3,
		b = 6,
		enteries = 3, 
		n = 30, 
		values = 15; 

	double d = 0.1,
		Em = 0.00001, 
		E, 
		T = 1; 

	double* W = new double[enteries]; 
	for (int i = 0; i < enteries; i++) { 
		W[i] = (double)(rand()) / RAND_MAX; 
		cout << "W[" << i << "] = " << W[i] << endl; 
	}
	cout << endl;

	double* etalon_values = new double[n + values]; 
	for (int i = 0; i < n + values; i++) { 
		double step = 0.1; 
		double x = step * i;
		etalon_values[i] = a * sin(b * x) + d; 
	}
	int era = 0; 

	while (1) {
		double y1; 
		double Alpha = 0.05; 
		E = 0; 

		for (int i = 0; i < n - enteries; i++) {
			y1 = 0;

			double temp = 0.0;
			for (int j = 0; j < enteries; j++) {
				temp += pow(etalon_values[i + j], 2);
			}
			Alpha = 1 / (1 + temp); 


			for (int j = 0; j < enteries; j++) { 
				y1 += W[j] * etalon_values[j + i];
			}
			y1 -= T;

			for (int j = 0; j < enteries; j++) { 
				W[j] -= Alpha * (y1 - etalon_values[i + enteries]) * etalon_values[i + j];
			}

			T += Alpha * (y1 - etalon_values[i + enteries]); 
			E += 0.5 * pow(y1 - etalon_values[i + enteries], 2);
			era++;

		}

		cout << era << " | " << E << endl;
		if (E < Em) break;
	}  
	cout << endl;

	cout << "РЕЗУЛЬТАТЫ ОБУЧЕНИЯ" << endl;
	cout << setw(27) << right << "Эталонные значения" << setw(23) << right << "Полученные значения";
	cout << setw(23) << right << "Отклонение" << endl;
	double* prognoz_values = new double[n + values];

	for (int i = 0; i < n; i++) {
		prognoz_values[i] = 0;
		for (int j = 0; j < enteries; j++) {
			prognoz_values[i] += W[j] * etalon_values[j + i]; 
		}
		prognoz_values[i] -= T;

		cout << "y[" << i + 1 << "] = " << setw(20) << right << etalon_values[i + enteries] << setw(23) << right;
		cout << prognoz_values[i] << setw(23) << right << etalon_values[i + enteries] - prognoz_values[i] << endl;
	}

	cout << endl << "РЕЗУЛЬТАТЫ ПРОГНОЗИРОВАНИЯ" << endl;
	cout << setw(28) << right << "Эталонные значения" << setw(23) << right << "Полученные значения" << setw(23) << right << "Отклонение" << endl;

	for (int i = 0; i < values; i++) {
		prognoz_values[i + n] = 0;
		for (int j = 0; j < enteries; j++) {
			prognoz_values[i + n] += W[j] * etalon_values[n - enteries + j + i];
		}
		prognoz_values[i + n] -= T;

		cout << "y[" << n + i + 1 << "] = " << setw(20) << right << etalon_values[i + n] << setw(23) << right;
		cout << prognoz_values[i + n] << setw(23) << right << etalon_values[i + n] - prognoz_values[i + n] << endl;
	}

	delete[]etalon_values;
	delete[]prognoz_values;
	delete[]W;

	system("pause");
	return 0;
}