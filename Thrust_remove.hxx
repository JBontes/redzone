#pragma once
#include <cuda.h>
#include "device_launch_parameters.h"
#include <thrust/remove.h>
#include <thrust/execution_policy.h>
#include <concepts>
#include <chrono>
#include "DataStruct.hxx"






__host__ void thrust_remove_phase2(int* list, int listsize) {
    thrust::remove(thrust::device, list, list + listsize, removeme);
}

void test_thrust(const int run) {
    constexpr auto maxelements = 1 << 29;
    TestLists_t TestList(maxelements, maxelements);
    auto list = TestList.dev_A;
    auto removals = TestList.dev_R;
    const int rp[] = { 100, -1, 0, 25, 50, 75 };

    //64 KiB = 16K x 4 -> 2^14 x 4;
    for (auto redzone_removals = 0; redzone_removals < 6; redzone_removals++) {
        auto redzone_percentage = -2;
        redzone_percentage = rp[redzone_removals];
            
        assert(redzone_percentage != -2);
        for (auto r = 0; r < 4; r++) {
            auto kpercentage = 100;
            switch (r) {
                case 0:  kpercentage = 2;  break;
                case 1:  kpercentage = 10; break;
                case 2:  kpercentage = 50; break;
                default: kpercentage = 90; break;
            }
            for (auto e = 29; e >= 14; e--) {
                const auto listsize = 1ull << e;
                const auto k = (listsize * kpercentage) / 100;
                //const auto greensize = listsize - k;
                //const auto redstart = greensize;

                if (kpercentage > 50 && redzone_percentage != -1) { continue; } //skip redzone investigations for 90%
                //Init the data
                TestList.Update(listsize, k, redzone_percentage);
                
                const auto start1 = std::chrono::high_resolution_clock::now();
                remove_phase1 <<<GPUBlockCount, GPUBlockSize>>> (list, removals, k);
                CUDA(cudaDeviceSynchronize());
                const auto end1 = std::chrono::high_resolution_clock::now();
                thrust_remove_phase2(list, listsize);
                CUDA(cudaDeviceSynchronize());
                const auto end2 = std::chrono::high_resolution_clock::now();
                const auto is_OK = TestList.isOK();
                const auto time1 = int(std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count());
                const auto time2 = int(std::chrono::duration_cast<std::chrono::microseconds>(end2 - end1).count());
                const auto status = is_OK ? "Pass" : "##### error!!!!!";
                printf("n = 1 << %i, k = %i%%, r = %i%%, time1 = %ius, time2 = %ius, OK = %s, run = %i\n", e, kpercentage, redzone_percentage, time1, time2, status, run);
            }
        }
    }
}