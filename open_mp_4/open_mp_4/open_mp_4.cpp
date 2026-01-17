#define NOMINMAX

#include <iostream>
#include <omp.h>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <windows.h>
#include <limits>
#include <iomanip>
#include <map>
#include <algorithm>

using namespace std;

int find_maxmin(const vector<vector<int>>& matrix, int num_threads) {
    int n = matrix.size();
    int max_of_mins = numeric_limits<int>::min();

#pragma omp parallel num_threads(num_threads)
    {
        int local_max_of_mins = numeric_limits<int>::min();

#pragma omp for
        for (int i = 0; i < n; i++) {
            int min_in_row = *min_element(matrix[i].begin(), matrix[i].end());
            if (min_in_row > local_max_of_mins) {
                local_max_of_mins = min_in_row;
            }
        }

#pragma omp critical
        {
            if (local_max_of_mins > max_of_mins) {
                max_of_mins = local_max_of_mins;
            }
        }
    }

    return max_of_mins;
}


int main() {

    SetConsoleOutputCP(1251);
    SetConsoleCP(1251);

    ofstream output("result_omp4.csv");

    output << "Threads,Size,Time(ms),Result,Speedup,Efficiency(%)\n";

    srand(static_cast<unsigned int>(time(0)));

    vector<int> sizes = { 1000, 2000, 5000, 10000 };
    vector<int> threads = { 1, 2, 4, 8, 16 };

    map<int, double> base_times;

    for (int size : sizes) {
        cout << "Размер матрицы: " << size << "x" << size << endl;

        vector<vector<int>> matrix(size, vector<int>(size));

#pragma omp parallel for num_threads(8)
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                matrix[i][j] = rand() % 10000;  
            }
        }

        for (int t : threads) {
            const int repetitions = 5;  
            double total_time = 0.0;
            int final_result = 0;

            for (int rep = 0; rep < repetitions; rep++) {
                double start = omp_get_wtime();
                int result = find_maxmin(matrix, t);
                double end = omp_get_wtime();
                total_time += (end - start) * 1000.0;  

                if (rep == 0) final_result = result;  
            }

            double avg_time = total_time / repetitions;

            if (t == 1) {
                base_times[size] = avg_time;
            }

            double base_time = base_times[size];
            double speedup = base_time / avg_time;
            double efficiency = (speedup / t) * 100.0;

            output << t << "," << size << "," << fixed << setprecision(3) << avg_time
                << "," << final_result << "," << speedup << "," << efficiency << "\n";

            cout << "   Потоков: " << setw(2) << t
                << "    Время: " << setw(10) << avg_time << " мс"
                << "    Ускорение: " << setw(6) << fixed << setprecision(2) << speedup << "x"
                << "    Эффективность: " << setw(6) << efficiency << "%"
                << "    Максимин: " << final_result << endl;
        }
        cout << endl;
    }

    output.close();

    return 0;
}
