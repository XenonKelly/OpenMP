#define NOMINMAX

#include <iostream>
#include <omp.h>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <cmath>
#include <windows.h>
#include <map>
#include <string>
#include <iomanip>
#include <limits>

using namespace std;

void uneven_workload(int iteration, int vector_size) {
    vector<double> vec(vector_size);
    double result = 0.0;

    int workload_type = iteration % 10;

    if (workload_type < 3) {
        for (int i = 0; i < vector_size; i++) {
            vec[i] = sin(i * 0.1) * cos(i * 0.1);
        }
        for (int i = 0; i < vector_size / 10; i++) {
            result += vec[i];
        }
    }
    else if (workload_type < 7) {
        for (int i = 0; i < vector_size; i++) {
            vec[i] = sqrt(i + 1.0) * log(i + 2.0);
        }
        for (int i = 0; i < vector_size / 5; i++) {
            result += vec[i] * vec[vector_size - i - 1];
        }
    }
    else {
        for (int i = 0; i < vector_size; i++) {
            vec[i] = 0.0;
            for (int j = 0; j < 3; j++) {
                vec[i] += sin(i * 0.01 + j) * cos(i * 0.01 - j);
            }
        }

        int sort_size = min(100, vector_size / 5);
        for (int i = 0; i < sort_size; i++) {
            for (int j = 0; j < sort_size - 1; j++) {
                if (vec[j] > vec[j + 1]) {
                    double temp = vec[j];
                    vec[j] = vec[j + 1];
                    vec[j + 1] = temp;
                }
            }
        }

        for (int i = 0; i < vector_size / 10; i++) {
            result += vec[i];
        }
    }

    (void)result;
}

double test_schedule(const string& schedule_type, int num_iterations, int num_threads, int vector_size) {
    double start_time = omp_get_wtime();

    if (schedule_type == "static") {
#pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int i = 0; i < num_iterations; i++) {
            uneven_workload(i, vector_size);
        }
    }
    else if (schedule_type == "dynamic") {
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, 10)
        for (int i = 0; i < num_iterations; i++) {
            uneven_workload(i, vector_size);
        }
    }
    else if (schedule_type == "guided") {
#pragma omp parallel for num_threads(num_threads) schedule(guided, 10)
        for (int i = 0; i < num_iterations; i++) {
            uneven_workload(i, vector_size);
        }
    }

    return (omp_get_wtime() - start_time) * 1000.0;
}

int main() {

    SetConsoleOutputCP(1251);
    SetConsoleCP(1251);

    ofstream output("result_omp6.csv");
    output << "Schedule_Type,Vector_Size,Iterations,Threads,Time(ms),Speedup,Efficiency(%)\n";

    srand((unsigned int)time(0));

    vector<int> vector_sizes = { 1000, 5000, 10000 };
    vector<int> iterations_list = { 200 };  
    vector<int> threads_list = { 1, 2, 4, 8, 16 };
    vector<string> schedules = { "static", "dynamic", "guided" };

    map<pair<string, int>, double> base_times;

    cout << "Однопоточный случай:" << endl;
    for (int vector_size : vector_sizes) {
        for (const string& schedule : schedules) {
            const int repetitions = 5;
            double total_time = 0.0;

            for (int rep = 0; rep < repetitions; rep++) {
                double time_ms = test_schedule(schedule, iterations_list[0], 1, vector_size);
                total_time += time_ms;
            }

            double avg_time = total_time / repetitions;
            base_times[{schedule, vector_size}] = avg_time;

            cout << "   Размер: " << setw(6) << vector_size
                << "    Стратегия: " << setw(8) << schedule
                << "    Время: " << setw(8) << avg_time << " мс" << endl;
        }
    }


    for (int vector_size : vector_sizes) {
        cout << "\nРазмер вектора: " << vector_size << endl;

        for (int threads : threads_list) {
            if (threads == 1) continue;

            for (const string& schedule : schedules) {
                const int repetitions = 5;
                double total_time = 0.0;

                for (int rep = 0; rep < repetitions; rep++) {
                    double time_ms = test_schedule(schedule, iterations_list[0], threads, vector_size);
                    total_time += time_ms;
                }

                double avg_time = total_time / repetitions;
                double base_time = base_times[{schedule, vector_size}];
                double speedup = base_time / avg_time;
                double efficiency = (speedup / threads) * 100.0;

                output << schedule << "," << vector_size << "," << iterations_list[0]
                    << "," << threads << "," << fixed << setprecision(3) << avg_time
                    << "," << speedup << "," << efficiency << "\n";

                cout << "   Потоков: " << setw(2) << threads
                    << "    Стратегия: " << setw(8) << schedule
                    << "    Время: " << setw(8) << avg_time << " мс"
                    << "    Ускорение: " << setw(5) << fixed << setprecision(2) << speedup << "x"
                    << "    Эффективность: " << setw(5) << efficiency << "%" << endl;
            }
        }
    }

    output.close();

    return 0;
}
