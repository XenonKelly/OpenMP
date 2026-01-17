#include <iostream>
#include <omp.h>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <map>
#include <windows.h>

using namespace std;

void scalar_product(const vector<int>& vec1, const vector<int>& vec2, long long& result, int num_threads) {
    result = 0;

#pragma omp parallel num_threads(num_threads)
    {
        long long local_sum = 0;

#pragma omp for
        for (int i = 0; i < vec1.size(); i++) {
            local_sum += (long long)vec1[i] * vec2[i];
        }

#pragma omp atomic
        result += local_sum;
    }
}

int main() {
    SetConsoleOutputCP(1251);
    SetConsoleCP(1251);

    ofstream output("results_omp2.csv");

    output << "Threads,Size,Time(ms),Speedup,Efficiency(%)\n";

    srand(static_cast<unsigned int>(time(0)));

    vector<int> sizes = { 500000, 1000000, 5000000, 10000000 };
    vector<int> threads = { 1, 2, 4, 8, 16 };

    map<int, double> base_times;


    for (int size : sizes) {
        cout << "\nРазмер векторов: " << size << endl;
        vector<int> vec1(size);
        vector<int> vec2(size);

        for (int i = 0; i < size; i++) {
            vec1[i] = rand() % 1000;  
            vec2[i] = rand() % 1000;
        }

        for (int t : threads) {
            const int repetitions = 10;  
            double total_time = 0.0;

            for (int rep = 0; rep < repetitions; rep++) {
                long long result;
                double start = omp_get_wtime();

                scalar_product(vec1, vec2, result, t);

                double end = omp_get_wtime();
                total_time += (end - start) * 1000.0;  
            }

            double avg_time = total_time / repetitions;

            if (t == 1) {
                base_times[size] = avg_time;
            }

            double base_time = base_times[size];
            double speedup = base_time / avg_time;
            double efficiency = (speedup / t) * 100.0;

            output << t << "," << size << "," << fixed << setprecision(3) << avg_time
                << "," << speedup << "," << efficiency << "\n";

            cout << "   Потоков: " << setw(2) << t
                << "    Время: " << setw(8) << avg_time << " мс"
                << "    Ускорение: " << setw(6) << speedup << "x"
                << "    Эффективность: " << setw(6) << efficiency << "%" << endl;
        }
    }

    output.close();

    return 0;
}
