#include <iostream>
#include <vector>
#include <chrono>
#include <fmt/core.h>
#include <omp.h>

namespace ch=std::chrono;

float sum_reduction(std::vector<float> input, const int n) {
    float suma = 0.0f;
    for (int i = 0; i < n; ++i)
        suma = suma + input[i];
    return suma;
}

float sum_omp_reduction(std::vector<float> input, const int n) {
    for (int i = n / 2; i > 0; i/=2) {
        #pragma omp parallel num_threads(i) default(none) shared(input)
        {
            int thread_id = omp_get_thread_num();
            int num_thread = omp_get_num_threads();

            float suma = input[thread_id] + input[thread_id + num_thread];
            input[thread_id] = suma;
        }
    }
    return input[0];
}

#define ARRAY_SIZE 1024
int main() {
    std::vector<float> data(ARRAY_SIZE, 0);

    for (int i = 0; i < data.size(); ++i) {
        data[i] = i;
    }

    {
        auto start = ch::high_resolution_clock::now();
        float suma_serial = sum_reduction(data, ARRAY_SIZE);
        auto end = ch::high_resolution_clock::now();
        ch::duration<double, std::milli> duration = end-start;
        fmt::println("Suma Serial: {}, tiempo: {}ms", suma_serial, duration.count());
    }

    {
        auto start = ch::high_resolution_clock::now();
        float suma_concurrente = sum_omp_reduction(data, ARRAY_SIZE);
        auto end = ch::high_resolution_clock::now();
        ch::duration<double, std::milli> duration = end-start;
        fmt::println("Suma Serial: {}, tiempo: {}ms", suma_concurrente, duration.count());
    }

    return 0;
}
