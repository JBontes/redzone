#pragma once
#include "Macros.hxx"
#include <random>
#include <cassert>
#include <algorithm>
#include <set>


static constexpr auto removeme = -1;

__global__ void remove_phase1(int* list, const int* removals, const int removecount) {
    const auto start = (blockIdx.x * blockDim.x) + threadIdx.x;
    const auto gridsize = (blockDim.x * gridDim.x);
    for (auto i = start; i < removecount; i += gridsize) {
        const auto removedest = removals[i];
        list[removedest] = removeme;
    }
}

struct TestLists_t {
    int* dev_A;
    int* dev_out;
    int* dev_R;
    //maximum device size: 3x2GiB = 6GiB.
    int* dev_Asize;
    int* dev_Rsize;
    mutable std::vector<bool> used;
    //maximum host size: 2x2GiB + 1/8x2GiB = 2.25 GiB
    float* dev_TestTime;
    int Asize;
    int Rsize;
    int* host_A;
    int* host_R;

    int* dev_A2;
    int* dev_R2;
    int* dev_Asize2;
    int* dev_Rsize2;
    float* dev_TestTime2; //phase 1..3 + total
private:
    void allocate() {
        assert(Asize >= Rsize);
        assert(Asize > 0);
        assert(Rsize > 0);
        host_A = (int*)malloc(Asize * sizeof(int));
        host_R = (int*)malloc(Rsize * sizeof(int));
        assert(nullptr != host_A);
        assert(nullptr != host_R);

        CUDA(cudaMalloc(&dev_A, Asize * sizeof(int)));
        CUDA(cudaMalloc(&dev_out, Asize * sizeof(int)));
        CUDA(cudaMalloc(&dev_R, Rsize * sizeof(int)));
        CUDA(cudaMalloc(&dev_Asize, sizeof(int)));
        CUDA(cudaMalloc(&dev_Rsize, sizeof(int)));
        CUDA(cudaMalloc(&dev_TestTime, sizeof(float) * 4)); //3 phases and the total time
        //printf("Allocation done\n");
        dev_A2 = dev_A;
        dev_R2 = dev_R;
        dev_Asize2 = dev_Asize;
        dev_Rsize2 = dev_Rsize;
        dev_TestTime2 = dev_TestTime;
    }

public:
    TestLists_t() = delete;
    TestLists_t(const int Asize, const int Rsize) : Asize(Asize), Rsize(Rsize) {
        allocate();
        Update(0, 0, 0); //do not fill the random data just yet
    }

    void Update(const int n, const int k) { assert(n > k);  Update(n, k, -1); }

    void Update(const int n, const int k, const int red_percentage) { 
        assert(n >= k);
        Asize = n;
        Rsize = k;
        for (auto i = 0; i < n; i++) { host_A[i] = i; } //increasing numbers, used to verify correctness

        used = std::vector<bool>(Asize, false);
        std::random_device rd;  // a seed source for the random number engine
        std::mt19937 rng(rd()); // mersenne_twister_engine seeded with rd()
        

        // Use distrib to transform the random unsigned int
        // generated by gen into an int in [1, 6]
        if (-1 == red_percentage) {
            auto range = n;
            std::vector<int> unique(&host_A[0], &host_A[range]);
            for (auto i = 0; i < k; i++) {
                std::uniform_int_distribution distrib(0, --range); //very fast, just sets 2 ints
                const auto r = distrib(rng);
                const auto item = unique[r];
                unique[r] = unique[range]; //fill the hole
                host_A[item] = removeme;
                host_R[i] = item;
            }
        } else {
            const auto red_k = (int64_t(k) * red_percentage) / 100;
            const auto green_k = k - red_k;

            //The greenzone runs from 0..n-k
            std::vector<int> uniquegreen(&host_A[0], &host_A[n-k]); 
            std::vector<int> uniquered(&host_A[n-k], &host_A[n]);
            auto range = n - k;
            for (auto i = 0; i < green_k; i++) {
                std::uniform_int_distribution distrib(0, --range); //very fast, just sets 2 ints
                const auto r = distrib(rng);
                const auto item = uniquegreen[r];
                uniquegreen[r] = uniquegreen[range]; //fill the hole
                host_A[item] = removeme;
                host_R[i] = item;
            }
            range = k;
            for (auto i = green_k; i < k; i++) {
                std::uniform_int_distribution distrib(0, --range); //very fast, just sets 2 ints
                const auto r = distrib(rng);
                const auto item = uniquered[r];
                uniquered[r] = uniquered[range]; //fill the hole
                host_A[item] = removeme;
                host_R[i] = item;
            }
        } //else
        used = std::vector<bool>(Asize, false);
        //for (auto i = 0; i < k; i++) { host_A[host_R[start_k + i]] = -1; }
        if (dev_A != dev_A2) { printf("dev_A != dev_A2"); }
        if (dev_R != dev_R2) { printf("dev_A != dev_A2"); }
        if (dev_Asize != dev_Asize2) { printf("dev_A != dev_A2"); }
        if (dev_Rsize != dev_Rsize2) { printf("dev_A != dev_A2"); }
        CUDA(cudaMemcpy(dev_A, host_A, Asize * sizeof(int), cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(dev_R, host_R, Rsize * sizeof(int), cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(dev_Asize, &Asize, sizeof(int), cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(dev_Rsize, &Rsize, sizeof(int), cudaMemcpyHostToDevice));
    }

    ~TestLists_t() {
        //printf("*** freeing ***\n");
        CUDA(cudaFree(dev_A));
        CUDA(cudaFree(dev_R));
        CUDA(cudaFree(dev_out));
        CUDA(cudaFree(dev_Asize));
        CUDA(cudaFree(dev_Rsize));
        CUDA(cudaFree(dev_TestTime));
        free(host_A);
        free(host_R);
    }

    int* get_A() const {
        CUDA(cudaMemcpy(host_A, dev_A, Asize * sizeof(int), cudaMemcpyDeviceToHost));
        return host_A;
    }

    int newsize() const { return Asize - Rsize; }

    bool isOK_CPU() const {
        used = std::vector<bool>(Asize, false);
        auto errors = 0;
        if (dev_A != dev_A2) {
            printf("dev_A is corrupted, is %p, should be %p", dev_A, dev_A2);  errors++;
        }
        if (dev_R != dev_R2) {
            printf("dev_R is corrupted, is %p, should be %p", dev_R, dev_R2);  errors++;
        }
        if (dev_Asize != dev_Asize2) {
            printf("dev_Asize is corrupted, is %p, should be %p", dev_Asize, dev_Asize2);  errors++;
        }
        if (dev_A != dev_A2) {
            printf("dev_Rsize is corrupted, is %p, should be %p", dev_Rsize, dev_Rsize2);  errors++;
        }
        if (dev_TestTime != dev_TestTime2) {
            printf("dev_TestTime is corrupted, is %p, should be %p", dev_TestTime, dev_TestTime2);  errors++;
        }
        auto error_minus = 0;
        auto error_big = 0;
        auto error_double = 0;
        for (auto i = 0; i < (Asize - Rsize); i++) {
            const auto val = host_A[i];
            if (val < 0) {
                error_minus++;
            } else if (val > Asize) {
                error_big++;
            } else if (used[val]) {
                error_double++;
            } else {
                used[val] = true;
            }
        }
        errors = error_minus + error_big + error_double;
        if (errors > 0) { printf(" !!!!!!!! There were %i errors: n = %i, k = %i, min = %i, big = %i, double = %i\n", errors, Asize, Rsize, error_minus, error_big, error_double); }
        used = std::vector<bool>(Asize, false);
        return (errors == 0);
    }

    bool isOK() const {
        //Test if removal went OK
        get_A();
        return isOK_CPU();
    }
};