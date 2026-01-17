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

vector<double> generate_data(int size) {
    vector<double> data(size);
    for (int i = 0; i < size; i++) {
        data[i] = (rand() % 1000) / 10.0;  
    }
    return data;
}

double reduction_atomic(const vector<double>& data, int num_threads) {
    double sum = 0.0;
    double start_time = omp_get_wtime();

#pragma omp parallel num_threads(num_threads)
    {
        double local_sum = 0.0;

#pragma omp for
        for (int i = 0; i < data.size(); i++) {
            local_sum += data[i];
        }

#pragma omp atomic
        sum += local_sum;
    }

    double end_time = omp_get_wtime();
    return (end_time - start_time) * 1000.0;
}

double reduction_critical(const vector<double>& data, int num_threads) {
    double sum = 0.0;
    double start_time = omp_get_wtime();

#pragma omp parallel num_threads(num_threads)
    {
        double local_sum = 0.0;

#pragma omp for
        for (int i = 0; i < data.size(); i++) {
            local_sum += data[i];
        }

#pragma omp critical
        {
            sum += local_sum;
        }
    }

    double end_time = omp_get_wtime();
    return (end_time - start_time) * 1000.0;
}

double reduction_lock(const vector<double>& data, int num_threads) {
    double sum = 0.0;
    omp_lock_t lock;
    omp_init_lock(&lock);

    double start_time = omp_get_wtime();

#pragma omp parallel num_threads(num_threads)
    {
        double local_sum = 0.0;

#pragma omp for
        for (int i = 0; i < data.size(); i++) {
            local_sum += data[i];
        }

        omp_set_lock(&lock);
        sum += local_sum;
        omp_unset_lock(&lock);
    }

    double end_time = omp_get_wtime();
    omp_destroy_lock(&lock);

    return (end_time - start_time) * 1000.0;
}

double reduction_builtin(const vector<double>& data, int num_threads) {
    double sum = 0.0;
    double start_time = omp_get_wtime();

#pragma omp parallel for reduction(+:sum) num_threads(num_threads)
    for (int i = 0; i < data.size(); i++) {
        sum += data[i];
    }

    double end_time = omp_get_wtime();
    return (end_time - start_time) * 1000.0;
}

int main() {

    SetConsoleOutputCP(1251);
    SetConsoleCP(1251);

    ofstream output("result_omp7.csv");

    output << "Method,Data_Size,Threads,Time(ms),Speedup,Efficiency(%),Result\n";

    srand(static_cast<unsigned int>(time(0)));

    vector<int> data_sizes = { 100000, 500000, 1000000, 5000000 };
    vector<int> threads_list = { 1, 2, 4, 8, 16 };

    vector<string> method_names = { "atomic", "critical", "lock", "reduction" };
    vector<string> method_names_ru = { "Атомарные операции", "Критические секции", "Замки", "Встроенная редукция" };

    map<pair<string, int>, double> base_times;

    cout << "Однопоточный случай:" << endl;

    for (int data_size : data_sizes) {
        vector<double> data = generate_data(data_size);

        double reference_sum = 0.0;
        for (double val : data) reference_sum += val;

        for (size_t method_idx = 0; method_idx < method_names.size(); method_idx++) {
            const string& method = method_names[method_idx];
            const int repetitions = 7;  

            double total_time = 0.0;

            for (int rep = 0; rep < repetitions; rep++) {
                double time_ms = 0.0;

                if (method == "atomic") {
                    time_ms = reduction_atomic(data, 1);
                }
                else if (method == "critical") {
                    time_ms = reduction_critical(data, 1);
                }
                else if (method == "lock") {
                    time_ms = reduction_lock(data, 1);
                }
                else if (method == "reduction") {
                    time_ms = reduction_builtin(data, 1);
                }

                total_time += time_ms;
            }

            double avg_time = total_time / repetitions;
            base_times[{method, data_size}] = avg_time;

            cout << "   Размер: " << setw(8) << data_size
                << "    Метод: " << setw(25) << method_names_ru[method_idx]
                << "    Время: " << setw(8) << fixed << setprecision(2) << avg_time << " мс" << endl;
        }
        cout << endl;
    }

    int test_counter = 0;
    int total_tests = data_sizes.size() * threads_list.size() * method_names.size();

    for (int data_size : data_sizes) {
        cout << "\nРазмер данных: " << data_size << endl;

        vector<double> data = generate_data(data_size);

        for (int threads : threads_list) {
            if (threads == 1) continue;

            cout << "\n  Потоков: " << threads << endl;

            for (size_t method_idx = 0; method_idx < method_names.size(); method_idx++) {
                test_counter++;
                const string& method = method_names[method_idx];

                const int repetitions = 7;
                double total_time = 0.0;
                double final_result = 0.0;

                for (int rep = 0; rep < repetitions; rep++) {
                    double time_ms = 0.0;
                    double result = 0.0;

                    if (method == "atomic") {
                        time_ms = reduction_atomic(data, threads);
                    }
                    else if (method == "critical") {
                        time_ms = reduction_critical(data, threads);
                    }
                    else if (method == "lock") {
                        time_ms = reduction_lock(data, threads);
                    }
                    else if (method == "reduction") {
                        time_ms = reduction_builtin(data, threads);
                    }

                    total_time += time_ms;

                    if (rep == 0) {
                        result = 0.0;
                        for (double val : data) result += val;
                        final_result = result;
                    }
                }

                double avg_time = total_time / repetitions;
                double base_time = base_times[{method, data_size}];
                double speedup = base_time / avg_time;
                double efficiency = (speedup / threads) * 100.0;

                output << method << "," << data_size << "," << threads << ","
                    << fixed << setprecision(3) << avg_time << ","
                    << speedup << "," << efficiency << ","
                    << scientific << setprecision(6) << final_result << "\n";

                cout << "    " << setw(25) << left << method_names_ru[method_idx]
                    << "    Время: " << setw(8) << fixed << setprecision(2) << avg_time << " мс"
                    << "    Ускорение: " << setw(5) << fixed << setprecision(2) << speedup << "x"
                    << "    Эффективность: " << setw(5) << efficiency << "%" << endl;
            }
        }
    }

    output.close();

    return 0;
}
