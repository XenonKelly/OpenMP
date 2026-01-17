#define _USE_MATH_DEFINES 

#include <iostream>
#include <omp.h>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <windows.h>

using namespace std;

double calculate_integral(double a, double b, int n, int num_threads) {
    double h = (b - a) / n;
    double sum = 0.0;

#pragma omp parallel for reduction(+:sum) num_threads(num_threads)
    for (int i = 0; i < n; i++) {
        double x = a + i * h;  
        sum += sin(x) * sin(x);
    }

    return sum * h;
}


int main() {

    SetConsoleOutputCP(1251);
    SetConsoleCP(1251);

    ofstream output("result_omp3.csv");

    output << "Threads,Intervals,Time(ms),Result,Speedup,Efficiency(%)\n";

    double a = 0.0;
    double b = M_PI;  

    vector<long long> intervals = { 1000000, 5000000, 10000000, 50000000 };
    vector<int> threads = { 1, 2, 4, 8, 16 };

    map<long long, double> base_times;

    for (long long n : intervals) {
        cout << "Количество интервалов: " << n << endl;

        for (int t : threads) {
            const int repetitions = 5;  
            double total_time = 0.0;
            double final_result = 0.0;

            for (int rep = 0; rep < repetitions; rep++) {
                double start = omp_get_wtime();
                double result = calculate_integral(a, b, n, t);
                double end = omp_get_wtime();
                total_time += (end - start) * 1000.0;  

                if (rep == 0) final_result = result;  
            }

            double avg_time = total_time / repetitions;

            if (t == 1) {
                base_times[n] = avg_time;
            }

            double base_time = base_times[n];
            double speedup = base_time / avg_time;
            double efficiency = (speedup / t) * 100.0;

            output << t << "," << n << "," << fixed << setprecision(3) << avg_time
                << "," << scientific << setprecision(10) << final_result
                << "," << fixed << setprecision(3) << speedup
                << "," << efficiency << "\n";

            cout << "   Потоков: " << setw(2) << t
                << "    Время: " << setw(10) << avg_time << " мс"
                << "    Ускорение: " << setw(6) << speedup << "x"
                << "    Эффективность: " << setw(6) << efficiency << "%"
                << "    Результат: " << scientific << setprecision(6) << final_result << endl;
        }
        cout << endl;
    }

    output.close();

    double answer = (b - a) / 2.0 - (sin(2.0 * b) - sin(2.0 * a)) / 4.0;

    cout << "Решение: " << fixed << setprecision(10) << answer << endl;

    return 0;
}
