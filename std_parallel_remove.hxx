#pragma once
#include <algorithm>
#include <execution>
#include <concepts>
#include <chrono>
#include "DataStruct.hxx"



constexpr auto par = std::execution::par_unseq;


void par_remove_phase1(int* list, const int* removals, const int removecount) {
    std::for_each(par, removals, removals + removecount, [&](const int remove_index) {
        list[remove_index] = removeme;
    });
}

void thread_pool_phase1(int* list, const int* removals, const int removecount) {
    const auto threadcount = int(std::thread::hardware_concurrency() - 1);
    const auto stride = divup(removecount, threadcount);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, threadcount), [&](const tbb::blocked_range<size_t>& r) {
        const auto pid = int(r.begin());
        const auto start = stride * pid;
        const auto end = std::min(start + stride, removecount);
        //const auto len = end - start;
        for (auto i = start; i < end; i++) {
            const auto remove_index = removals[i];
            list[remove_index] = removeme;
        }
    });
}

void par_remove_phase2(int* list, const int listsize) {
    std::remove(par, list, list + listsize, removeme);
}

void test_par_remove(const int run) {
    constexpr auto maxelements = 1 << 29;
    TestLists_t TestList(maxelements, maxelements);
    auto list = TestList.host_A;
    auto removals = TestList.host_R;
    const int rp[] = { -1, 0, 25, 50, 75, 100 };

    //64 KiB = 16K x 4 -> 2^14 x 4;
    for (auto redzone_removals = 0; redzone_removals < 1; redzone_removals++) {
        auto redzone_percentage = rp[redzone_removals];

        for (auto r = 0; r < 4; r++) {
            auto kpercentage = 100;
            switch (r) {
                case 0:  kpercentage = 2;  break;
                case 1:  kpercentage = 10; break;
                case 2:  kpercentage = 50; break;
                default: kpercentage = 90; break;
            }
            for (auto e = 14; e <= 29; e++) {
                const auto listsize = 1ull << e;
                const auto k = (listsize * kpercentage) / 100;
                //const auto greensize = listsize - k;
                //const auto redstart = greensize;

                if (kpercentage > 50 && redzone_percentage != -1) { continue; } //skip redzone investigations for 90%
                //Init the data
                TestList.Update(listsize, k, redzone_percentage);
                //Make sure that phase 1 takes about the same amount of time it would with the thread pool
                //Spoiler: times are very close.
              
                const auto start1 = std::chrono::high_resolution_clock::now();
                par_remove_phase1(list, removals, k);
                const auto end1 = std::chrono::high_resolution_clock::now();
                par_remove_phase2(list, listsize);
                const auto end2 = std::chrono::high_resolution_clock::now();
                //const auto is_OK = TestList.isOK_CPU();
                const auto is_OK = true;
                //const auto time0 = int(std::chrono::duration_cast<std::chrono::microseconds>(end0 - start0).count());
                const auto time1 = int(std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count());
                const auto time2 = int(std::chrono::duration_cast<std::chrono::microseconds>(end2 - end1).count());
                const auto status = is_OK ? "Pass" : "##### error!!!!!";
                printf("n = 1 << %i, k = %i%%, r = %i%%, time1 = %i, time2 = %i, totaltime = %i, OK = %s, run = %i\n", e, kpercentage, redzone_percentage, time1, time2, time1 + time2, status, run);
            }
        }
    }
}