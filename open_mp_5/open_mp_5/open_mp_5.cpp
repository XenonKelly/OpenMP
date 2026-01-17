#define NOMINMAX  

#include <iostream>
#include <omp.h>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <windows.h>
#include <map>
#include <string>
#include <algorithm>
#include <iomanip>
#include <limits>

using namespace std;

enum MatrixType {
    DIAGONAL,      
    TRIANGULAR,    
    BANDED         
};


vector<vector<int>> create_special_matrix(int size, MatrixType type) {
    vector<vector<int>> matrix(size, vector<int>(size, 0));

    switch (type) {
    case DIAGONAL:
        for (int i = 0; i < size; i++) {
            matrix[i][i] = rand() % 100 + 1;
        }
        break;

    case TRIANGULAR:
        for (int i = 0; i < size; i++) {
            for (int j = i; j < size; j++) {
                matrix[i][j] = rand() % 100 + 1;
            }
        }
        break;

    case BANDED:
        for (int i = 0; i < size; i++) {
            int start = max(0, i - 2);
            int end = min(size - 1, i + 2);
            for (int j = start; j <= end; j++) {
                matrix[i][j] = rand() % 100 + 1;
            }
        }
        break;
    }
    return matrix;
}

int find_maximin_schedule(const vector<vector<int>>& matrix, int num_threads, const string& schedule_type, int chunk_size = 10) {
    int n = (int)matrix.size();
    int global_max_of_mins = numeric_limits<int>::min();

#pragma omp parallel num_threads(num_threads)
    {
        int local_max_of_mins = numeric_limits<int>::min();

        string effective_schedule = schedule_type;

        if (effective_schedule == "static") {
#pragma omp for schedule(static, chunk_size)
            for (int i = 0; i < n; i++) {
                int min_in_row = numeric_limits<int>::max();
                for (int j = 0; j < (int)matrix[i].size(); j++) {
                    if (matrix[i][j] != 0 && matrix[i][j] < min_in_row) {  
                        min_in_row = matrix[i][j];
                    }
                }
                if (min_in_row != numeric_limits<int>::max() && min_in_row > local_max_of_mins) {
                    local_max_of_mins = min_in_row;
                }
            }
        }
        else if (effective_schedule == "dynamic") {
#pragma omp for schedule(dynamic, chunk_size)
            for (int i = 0; i < n; i++) {
                int min_in_row = numeric_limits<int>::max();
                for (int j = 0; j < (int)matrix[i].size(); j++) {
                    if (matrix[i][j] != 0 && matrix[i][j] < min_in_row) {
                        min_in_row = matrix[i][j];
                    }
                }
                if (min_in_row != numeric_limits<int>::max() && min_in_row > global_max_of_mins) {
                    global_max_of_mins = min_in_row;
                }
            }
        }
        else if (effective_schedule == "guided") {
#pragma omp for schedule(guided, chunk_size)
            for (int i = 0; i < n; i++) {
                int min_in_row = numeric_limits<int>::max();
                for (int j = 0; j < (int)matrix[i].size(); j++) {
                    if (matrix[i][j] != 0 && matrix[i][j] < min_in_row) {
                        min_in_row = matrix[i][j];
                    }
                }
                if (min_in_row != numeric_limits<int>::max() && min_in_row > global_max_of_mins) {
                    global_max_of_mins = min_in_row;
                }
            }
        }
#pragma omp critical
        {
            if (local_max_of_mins > global_max_of_mins) {
                global_max_of_mins = local_max_of_mins;
            }
        }
    }

    return global_max_of_mins;
}

int main() {

    SetConsoleOutputCP(1251);
    SetConsoleCP(1251);

    ofstream output("result_omp5.csv");
    output << "Matrix_Type,Size,Threads,Strategy,Time(ms),Result,Speedup,Efficiency(%)\n";

    srand((unsigned int)time(0));

    vector<int> sizes = { 1000, 5000 };
    vector<int> threads = { 1, 2, 4, 8 };
    vector<string> schedules = { "static", "dynamic", "guided" };

    vector<string> type_names = { "Диагональная", "Треугольная", "Ленточная" };
    vector<string> type_names_en = { "diagonal", "triangular", "banded" };

    map<pair<string, int>, double> base_times;

    for (int type_idx = 0; type_idx <= BANDED; type_idx++) {
        MatrixType type = static_cast<MatrixType>(type_idx);

        cout << "\n" << type_names[type_idx] << " матрица" << endl;

        for (int size : sizes) {
            cout << "\n  Размер: " << size << "x" << size << endl;

            auto matrix = create_special_matrix(size, type);

            for (int t : threads) {
                for (const string& schedule : schedules) {
                    const int repetitions = 5;
                    double total_time = 0.0;
                    int final_result = 0;

                    for (int rep = 0; rep < repetitions; rep++) {
                        double start = omp_get_wtime();
                        int result = find_maximin_schedule(matrix, t, schedule);
                        double end = omp_get_wtime();
                        total_time += (end - start) * 1000.0;

                        if (rep == 0) final_result = result;
                    }

                    double avg_time = total_time / repetitions;

                    if (t == 1) {
                        base_times[{type_names_en[type_idx], size}] = avg_time;
                    }

                    double base_time = base_times[{type_names_en[type_idx], size}];
                    double speedup = (t == 1) ? 1.0 : (base_time / avg_time);
                    double efficiency = (speedup / t) * 100.0;

                    output << type_names_en[type_idx] << "," << size << "," << t << ","
                        << schedule << "," << fixed << setprecision(3) << avg_time << ","
                        << final_result << "," << speedup << "," << efficiency << "\n";

                    cout << "   Потоков: " << setw(2) << t
                        << "    Стратегия: " << setw(8) << schedule
                        << "    Время: " << setw(8) << avg_time << " мс"
                        << "    Ускорение: " << setw(5) << fixed << setprecision(2) << speedup << "x"
                        << "    Эффективность: " << setw(5) << efficiency << "%"
                        << "    Результат: " << final_result << endl;
                }
            }
        }
    }

    output.close();

    return 0;
}
