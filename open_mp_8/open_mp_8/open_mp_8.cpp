#define NOMINMAX

#include <iostream>
#include <omp.h>
#include <vector>
#include <fstream>
#include <windows.h>
#include <map>
#include <iomanip>
#include <string>
#include <cmath>
#include <queue>
#include <utility> 

using namespace std;

void generate_vector_file(const string& filename, int num_pairs, int vector_size) {
    ofstream file(filename, ios::binary);
    if (!file.is_open()) {
        cerr << "Ошибка создания файла " << endl;
        return;
    }

    file.write(reinterpret_cast<const char*>(&num_pairs), sizeof(int));
    file.write(reinterpret_cast<const char*>(&vector_size), sizeof(int));

    srand(static_cast<unsigned int>(time(0)));

    vector<double> vec(vector_size);
    for (int i = 0; i < num_pairs * 2; i++) { 
        for (int j = 0; j < vector_size; j++) {
            vec[j] = (rand() % 1000) / 100.0;
        }
        file.write(reinterpret_cast<const char*>(vec.data()), sizeof(double) * vector_size);
    }

    file.close();
    cout << "Файл " << filename << "сгенерирован\n";
}

bool load_pair(ifstream& file, vector<double>& vec1, vector<double>& vec2, int vector_size) {
    vec1.resize(vector_size);
    vec2.resize(vector_size);

    if (!file.read(reinterpret_cast<char*>(vec1.data()), sizeof(double) * vector_size)) {
        return false;
    }
    if (!file.read(reinterpret_cast<char*>(vec2.data()), sizeof(double) * vector_size)) {
        return false;
    }
    return true;
}

double compute_dot_product(const vector<double>& vec1, const vector<double>& vec2, int num_threads) {
    double sum = 0.0;
    int size = vec1.size();

    if (num_threads > 1 && size >= 1000) { 
#pragma omp parallel for reduction(+:sum) num_threads(num_threads)
        for (int i = 0; i < size; i++) {
            sum += vec1[i] * vec2[i];
        }
    }
    else {
        for (int i = 0; i < size; i++) {
            sum += vec1[i] * vec2[i];
        }
    }

    return sum;
}

double process_vectors_with_sections(const string& filename, int num_pairs, int vector_size, int num_threads) {
    double total_time = 0.0;

    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        cerr << "Ошибка открытия файла: " << filename << endl;
        return 0.0;
    }

    int file_pairs, file_size;
    file.read(reinterpret_cast<char*>(&file_pairs), sizeof(int));
    file.read(reinterpret_cast<char*>(&file_size), sizeof(int));

    int pairs_to_process = min(num_pairs, file_pairs);
    int size_to_process = min(vector_size, file_size);

    vector<double> results(pairs_to_process, 0.0);
    double total_sum = 0.0;

    double start_time = omp_get_wtime();

    if (num_threads == 1) {
        for (int i = 0; i < pairs_to_process; i++) {
            vector<double> vec1, vec2;
            if (load_pair(file, vec1, vec2, size_to_process)) {
                results[i] = compute_dot_product(vec1, vec2, num_threads);
                total_sum += results[i];
            }
        }
    }
    else {
        queue<pair<vector<double>, vector<double>>> q;
        vector<bool> processed(pairs_to_process, false); 

#pragma omp parallel sections num_threads(2)
        {
#pragma omp section
            {
                for (int i = 0; i < pairs_to_process; i++) {
                    vector<double> vec1, vec2;
                    if (load_pair(file, vec1, vec2, size_to_process)) {
#pragma omp critical
                        {
                            q.push({ vec1, vec2 });
                        }
                    }
                    else {
                        break;
                    }
                }
            }

#pragma omp section
            {
                int idx = 0;
                while (idx < pairs_to_process) {
                    pair<vector<double>, vector<double>> p;
                    bool got = false;

#pragma omp critical
                    {
                        if (!q.empty()) {
                            p = q.front();
                            q.pop();
                            got = true;
                        }
                    }

                    if (got) {
                        double dot = compute_dot_product(p.first, p.second, num_threads - 1); 
                        results[idx] = dot;
#pragma omp atomic
                        total_sum += dot;
                        idx++;
                    }
                    else {
                        Sleep(1); 
                    }
                }
            }
        }
    }

    double end_time = omp_get_wtime();
    total_time = end_time - start_time;

    file.close();

    cout << " [Сумма всех произведений: " << fixed << setprecision(2) << total_sum << "]";

    return total_time;
}

int main() {

    SetConsoleOutputCP(1251);
    SetConsoleCP(1251);

    const string filename = "vectors_data.bin"; 

    cout << "Генерация файла с векторами" << endl;
    generate_vector_file(filename, 1000, 1000000);  

    ofstream output("result_omp8.csv");
    output << "Pairs,Vector_Size,Threads,Time(sec),Speedup,Efficiency\n";

    vector<int> num_pairs_list = { 100, 500, 1000 };
    vector<int> vector_sizes = { 5000, 10000, 100000, 500000, 1000000 };
    vector<int> threads_list = { 1, 2, 4, 8 };


    map<pair<int, int>, double> base_times;

    for (int num_pairs : num_pairs_list) {
        for (int vector_size : vector_sizes) {
            cout << "\nПары: " << num_pairs << " Размер: " << vector_size << endl;

            for (int threads : threads_list) {
                double total_time = 0.0;
                const int repetitions = 3;

                for (int rep = 0; rep < repetitions; rep++) {
                    double time_sec = process_vectors_with_sections(filename, num_pairs, vector_size, threads);
                    total_time += time_sec;
                }

                double avg_time = total_time / repetitions;

                if (threads == 1) {
                    base_times[{num_pairs, vector_size}] = avg_time;
                }

                double speedup = (threads == 1) ? 1.0 : (base_times[{num_pairs, vector_size}] / avg_time);
                double efficiency = speedup / threads;

                output << num_pairs << "," << vector_size << "," << threads << ","
                    << fixed << setprecision(4) << avg_time << ","
                    << speedup << "," << efficiency << "\n";

                cout << "   Потоков: " << setw(2) << threads
                    << "    Время: " << setw(8) << avg_time << " сек"
                    << "    Ускорение: " << setw(5) << fixed << setprecision(2) << speedup << "x"
                    << "    Эффективность: " << setw(5) << fixed << setprecision(2) << efficiency << endl;
            }
        }
    }

    output.close();

    return 0;
}
