#include <iostream>
#include <omp.h>
#include <vector>
#include <climits>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <windows.h>
#include <iomanip>

using namespace std;

class ParallelMinMaxFinder {
private:
    vector<int> generate_test_data(int size) {
        vector<int> data(size);
        for (int i = 0; i < size; i++) {
            data[i] = (rand() * rand()) % 1000000;
        }
        return data;
    }

public:
    int find_max_with_reduction(const vector<int>& data, int threads) {
        int max_val = INT_MIN;
#pragma omp parallel for reduction(max:max_val) num_threads(threads)
        for (int i = 0; i < data.size(); i++) {
            if (data[i] > max_val) max_val = data[i];
        }
        return max_val;
    }

    int find_max_manual_split(const vector<int>& data, int threads) {
        int chunk_size = data.size() / threads;
        vector<int> local_maxes(threads, INT_MIN);

#pragma omp parallel num_threads(threads)
        {
            int thread_id = omp_get_thread_num();
            int start = thread_id * chunk_size;
            int end = (thread_id == threads - 1) ? data.size() : start + chunk_size;

            for (int i = start; i < end; i++) {
                if (data[i] > local_maxes[thread_id]) {
                    local_maxes[thread_id] = data[i];
                }
            }
        }

        int global_max = INT_MIN;
        for (int i = 0; i < threads; i++) {
            if (local_maxes[i] > global_max) {
                global_max = local_maxes[i];
            }
        }
        return global_max;
    }

    int find_min_with_reduction(const vector<int>& data, int threads) {
        int min_val = INT_MAX;
#pragma omp parallel for reduction(min:min_val) num_threads(threads)
        for (int i = 0; i < data.size(); i++) {
            if (data[i] < min_val) min_val = data[i];
        }
        return min_val;
    }

    int find_min_manual_split(const vector<int>& data, int threads) {
        int chunk_size = data.size() / threads;
        vector<int> local_mins(threads, INT_MAX);

#pragma omp parallel num_threads(threads)
        {
            int thread_id = omp_get_thread_num();
            int start = thread_id * chunk_size;
            int end = (thread_id == threads - 1) ? data.size() : start + chunk_size;

            for (int i = start; i < end; i++) {
                if (data[i] < local_mins[thread_id]) {
                    local_mins[thread_id] = data[i];
                }
            }
        }

        int global_min = INT_MAX;
        for (int i = 0; i < threads; i++) {
            if (local_mins[i] < global_min) {
                global_min = local_mins[i];
            }
        }
        return global_min;
    }

    void finding() {
        ofstream out_max("max_result_omp1.csv");
        ofstream out_min("min_result_omp1.csv");

        out_max << "Size,Threads,Reduction_Time,Manual_Time,Speedup_Reduction,Efficiency_Reduction,Speedup_Manual,Efficiency_Manual\n";
        out_min << "Size,Threads,Reduction_Time,Manual_Time,Speedup_Reduction,Efficiency_Reduction,Speedup_Manual,Efficiency_Manual\n";

        srand(time(0));

        int sizes[] = { 100000, 250000, 500000, 1000000, 2500000, 5000000 };
        int thread_counts[] = { 1, 2, 4, 8, 16 };

        for (int size : sizes) {
            cout << "\nЭлементов: " << size << endl;
            vector<int> test_data = generate_test_data(size);

            double base_time_reduction_max = 0.0;
            double base_time_manual_max = 0.0;
            double base_time_reduction_min = 0.0;
            double base_time_manual_min = 0.0;

            cout << "\n  Поиск максимума:" << endl;
            for (int threads : thread_counts) {
                double start = omp_get_wtime();
                int max_reduction = find_max_with_reduction(test_data, threads);
                double time_reduction = (omp_get_wtime() - start) * 1000.0;

                start = omp_get_wtime();
                int max_manual = find_max_manual_split(test_data, threads);
                double time_manual = (omp_get_wtime() - start) * 1000.0;

                if (threads == 1) {
                    base_time_reduction_max = time_reduction;
                    base_time_manual_max = time_manual;
                }

                double speedup_reduction = (threads == 1) ? 1.0 : base_time_reduction_max / time_reduction;
                double efficiency_reduction = speedup_reduction / threads;

                double speedup_manual = (threads == 1) ? 1.0 : base_time_manual_max / time_manual;
                double efficiency_manual = speedup_manual / threads;

                out_max << size << "," << threads << ","
                    << fixed << setprecision(3) << time_reduction << ","
                    << time_manual << ","
                    << speedup_reduction << "," << efficiency_reduction << ","
                    << speedup_manual << "," << efficiency_manual << "\n";

                cout << "   Потоки: " << setw(2) << threads
                    << "    Время (редукция): " << setw(8) << time_reduction << " ms"
                    << "    Время (без редукции): " << setw(8) << time_manual << " ms"
                    << "    Ускорение (редукция): " << setw(6) << speedup_reduction
                    << "    Эффективность (редукция): " << setw(6) << efficiency_reduction
                    << "    Ускорение (ручной): " << setw(6) << speedup_manual
                    << "    Эффективность (ручной): " << setw(6) << efficiency_manual << endl;
            }

            cout << "\n  Поиск минимума:" << endl;
            for (int threads : thread_counts) {
                double start = omp_get_wtime();
                int min_reduction = find_min_with_reduction(test_data, threads);
                double time_reduction = (omp_get_wtime() - start) * 1000.0;

                start = omp_get_wtime();
                int min_manual = find_min_manual_split(test_data, threads);
                double time_manual = (omp_get_wtime() - start) * 1000.0;

                if (threads == 1) {
                    base_time_reduction_min = time_reduction;
                    base_time_manual_min = time_manual;
                }

                double speedup_reduction = (threads == 1) ? 1.0 : base_time_reduction_min / time_reduction;
                double efficiency_reduction = speedup_reduction / threads;

                double speedup_manual = (threads == 1) ? 1.0 : base_time_manual_min / time_manual;
                double efficiency_manual = speedup_manual / threads;

                out_min << size << "," << threads << ","
                    << fixed << setprecision(3) << time_reduction << ","
                    << time_manual << ","
                    << speedup_reduction << "," << efficiency_reduction << ","
                    << speedup_manual << "," << efficiency_manual << "\n";

                cout << "   Потоки: " << setw(2) << threads
                    << "    Время (редукция): " << setw(8) << time_reduction << " ms"
                    << "    Время (без редукции): " << setw(8) << time_manual << " ms"
                    << "    Ускорение (редукция): " << setw(6) << speedup_reduction
                    << "    Эффективность (редукция): " << setw(6) << efficiency_reduction
                    << "    Ускорение (ручной): " << setw(6) << speedup_manual
                    << "    Эффективность (ручной): " << setw(6) << efficiency_manual << endl;
            }
            cout << endl;
        }

        out_max.close();
        out_min.close();
    }
};

int main() {
    SetConsoleOutputCP(1251);
    SetConsoleCP(1251);

    ParallelMinMaxFinder finder;
    finder.finding();

    system("pause");
    return 0;
}
