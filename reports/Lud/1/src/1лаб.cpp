#include <iostream>
#include <iomanip>
#include <Windows.h>
using namespace std;

int main() {
    SetConsoleCP(1251);
    SetConsoleOutputCP(1251);

    int a = 4,
        b = 8,
        inputs_number = 3, // входы ИНС
        n_obychenie = 30, // кол зн на кот проводится обучение
        n_prognoz = 15, // кол зн на кот проводится прогнозирование
        era = 0;

    double d = 0.4,
           x = 0,
           h = 0.1, // шаг
           min_error = 0.001, //минимальная ошибка
           sum_error, //суммарная ошибка
           T = 1; // пороговое значение ИНС

    double* w = new double[inputs_number]; // весовые коэффициенты
    for (int i = 0; i < inputs_number; i++) { // задаём их с помощью рандома
        w[i] = rand() % 100 * 0.1;
    }

    double* etalon = new double[n_obychenie + n_prognoz]; // эланонные зн для у
    for (int i = 0; i < n_obychenie + n_prognoz; i++) { // вычисление эталонных значений
        x += h;
        etalon[i] = a * sin(b * x) + d;
    }

    double y, // вых зн ИНС
        V = 0.03; // скорость
    do {
        sum_error = 0;
        for (int i = 0; i < n_obychenie - inputs_number; i++) {
            y = 0;
            for (int j = 0; j < inputs_number; j++) {
                y += w[j] * etalon[i + j];
            }
            y -= T;
            for (int j = 0; j < inputs_number; j++) {
                w[j] -= V * (y - etalon[i + inputs_number]) * etalon[i + j];
            }
            sum_error += 0.5 * pow((y - etalon[i + inputs_number]), 2);
            T += V * (y - etalon[i + inputs_number]);
        }
        era++;
    } while (sum_error > min_error);

    cout << "Количество эпох: " << era << endl;
    cout << "||Результаты Обучения||" << endl;
    cout << setw(27) << left << "|Эталонные значения|" << setw(29) << left << "|Полученные значения|" << setw(30) << left << "|Отклонение|" << endl;

    double* prognoz = new double[n_obychenie + n_prognoz];
    for (int i = 0; i < n_obychenie; i++) { //значения в результате обучения
        prognoz[i] = 0;
        for (int j = 0; j < inputs_number; j++) {
            prognoz[i] += w[j] * etalon[i + j];
        }
        prognoz[i] -= T;

        cout << "y[" << i << "] = " << setw(25) << left << etalon[i + inputs_number] << setw(25) << left;
        cout << prognoz[i] << setw(30) << left << pow(etalon[i + inputs_number] - prognoz[i], 2) << endl;
    }

    cout << "||Результаты Прогнозирования||" << endl;
    cout << setw(27) << left << "|Эталонные значения|" << setw(29) << left << "|Полученные значения|" << setw(30) << left << "|Отклонение|" << endl;

    for (int i = 0; i < n_prognoz; i++) { //прогнозируемые значения
        prognoz[i + n_obychenie] = 0;
        for (int j = 0; j < inputs_number; j++) {
            prognoz[i + n_obychenie] += w[j] * etalon[n_obychenie + j + i - inputs_number];
        }
        prognoz[i + n_obychenie] -= T;

        cout << "y[" << n_obychenie + i << "] = " << setw(25) << left << etalon[i + n_obychenie] << setw(25) << left;
        cout << prognoz[i + n_obychenie] << setw(30) << left << pow(etalon[i + n_obychenie] - prognoz[i + n_obychenie], 2) << endl;
    }
    delete[] w;
    delete[]etalon;
    delete[]prognoz;
    system("pause");
    return 0;
}
