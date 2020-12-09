#include <iostream> 
using namespace std;

double Sigmoid(double x) {
	return 1 / (1 + pow(2.71828, -x));
}

double* Gethiddens(bool entrances[], double Wes1_2[20][40], double T_hid[]) {
	double* hidden = new double[40];
	for (int i = 0; i < 40; i++) {
		hidden[i] = 0;
	}
	for (int i = 0; i < 40; i++) {
		for (int j = 0; j < 20; j++) {
			hidden[i] += Wes1_2[j][i] * entrances[j];
		}
		hidden[i] -= T_hid[i];
		hidden[i] = Sigmoid(hidden[i]);
	}
	return hidden;
}

double* result(bool entrances[], double Wes1_2[20][40], double T_hid[], double Wes2_3[40][3], double T_out[], double hidden[40]) {
	double* Results = new double[3];
	for (int i = 0; i < 3; i++) {
		Results[i] = 0;
	}
	for (int j = 0; j < 3; j++) {
		for (int i = 0; i < 40; i++) {
			Results[j] += hidden[i] * Wes2_3[i][j];
		}
		Results[j] -= T_out[j];
		Results[j] = Sigmoid(Results[j]);
	}
	return Results;
}

int main() {
	setlocale(0,"");
	int eras = 0;
	bool Vect_1[] = { 1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0 };
	bool Vect_2[] = { 1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1 };
	bool Vect_3[] = { 1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1 };
	bool* entrances = new bool[20];
	for (int i = 0; i < 20; i++) {
		entrances[i] = 0;
	}
	bool** Vectors = new bool* [8];
	Vectors[0] = Vect_1;
	Vectors[1] = Vect_2;
	Vectors[2] = Vect_3;
	double Wes1_2[20][40], Wes2_3[40][3], T_hid[40], T_out[3], E_min = 0.001, alpha = 0.04, reference, E = 0, Outputs[3] = { 0 };
	double* currents = new double[3];
	double* hidden = new double[40];
	double errors[3] = { 0 };
	double references[3] = { 0 };
	double errorsHid[40] = { 0 };

	for (int i = 0; i < 20; i++) {
		for (int j = 0; j < 40; j++) {
			Wes1_2[i][j] = ((double)rand() / (RAND_MAX)) - 0.5;
			for (int k = 0; k < 3; k++) {
				Wes2_3[j][k] = ((double)rand() / (RAND_MAX)) - 0.5;
				T_out[k] = ((double)rand() / (RAND_MAX)) - 0.5;
			}
			T_hid[j] = ((double)rand() / (RAND_MAX)) - 0.5;
		}
	}
	do {
		E = 0;
		for (int N = 0; N < 3; N++) {
			references[0] = references[1] = references[2] = 0;
			references[N] = 1;
			entrances = Vectors[N];
			hidden = Gethiddens(entrances, Wes1_2, T_hid);
			currents = result(entrances, Wes1_2, T_hid, Wes2_3, T_out, hidden);
			for (int i = 0; i < 3; i++) {
				errors[i] = currents[i] - references[i];
			}
			for (int j = 0; j < 40; j++) {
				for (int m = 0; m < 3; m++) {
					errorsHid[j] += errors[m] * currents[m] * (1 - currents[m]) * Wes2_3[j][m];
				}
			}
			for (int j = 0; j < 3; j++) {
				for (int i = 0; i < 40; i++) {
					Wes2_3[i][j] -= alpha * errors[j] * currents[j] * (1 - currents[j]) * hidden[i];
				}
				T_out[j] += alpha * errors[j] * currents[j] * (1 - currents[j]);
			}

			for (int j = 0; j < 40; j++) {
				for (int i = 0; i < 20; i++) {
					Wes1_2[i][j] -= alpha * errorsHid[j] * hidden[j] * (1 - hidden[j]) * entrances[i];
				}
				T_hid[j] += alpha * errorsHid[j] * hidden[j] * (1 - hidden[j]);
			}
			E += pow(errors[N], 2);
		}
		E /= 2;
		cout << "\rError: " << E;
		eras++;
	} while (E > E_min);
	cout << endl;
	cout << eras << endl;

	double* HiddenPred;
	double* Values;
	bool Vectors3[] = { 0,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0 };
	bool Vectors4[] = { 0,0,1,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0 };
	bool Vectors5[] = { 0,0,1,0,1,1,0,0,0,1,0,0,1,1,0,0,1,1,0,0 };
	bool Vectors6[] = { 1,1,1,1,1,0,0,0,1,0,1,1,0,0,0,0,0,1,1,1 };
	bool Vectors7[] = { 0,1,1,1,1,1,0,0,1,1,1,1,0,0,0,0,1,1,1,1 };
	Vectors[3] = Vectors3;
	Vectors[4] = Vectors4;
	Vectors[5] = Vectors5;
	Vectors[6] = Vectors6;
	Vectors[7] = Vectors7;


	for (int i = 0; i < 8; i++) {
		entrances = Vectors[i];
		cout << "Vector " << i + 1 << ": ";
		for (int j = 0; j < 20; j++) {
			cout << entrances[j] << " ";
		}
		cout << endl;
		cout << "REsult: ";
		HiddenPred = Gethiddens(entrances, Wes1_2, T_hid);
		Values = result(entrances, Wes1_2, T_hid, Wes2_3, T_out, HiddenPred);
		cout << Values[0] << " " << Values[1] << " " << Values[2] << endl;
		cout << endl;
	}
}