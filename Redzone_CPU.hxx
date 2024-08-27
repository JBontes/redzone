#pragma once

#include "tbb/parallel_for.h"
#include <chrono>
#include <cassert>
#include <concepts>
#include <array>
#include "DataStruct.hxx"

static constexpr auto deleted = 0x8000'0000;

__host__ void phase1_mark_redzone(int* removals, const int start, const int end, const int redzonestart) {
    assert(sourceorphans.size() == threadcount);
    assert([&](){ for (auto i = 0; i < threadcount; i++) { if (sourceorphans[i] != 0) { return false } } return true; });
    auto orphancount = 0;
    for (auto i = start; i < end; i++) {
        const auto rz_index = (int(removals[i] &(~deleted))) - redzonestart;
        if (rz_index >= 0) {
            removals[rz_index] |= deleted;
            orphancount++;
        }
    } //for i
    //printf("T: %i, orphancount = %i\n", tid, orphancount);
}

__host__ void phase2_delete_and_collect_orphans(int* list, int* removals, const int start, const int end, const int redzonestart, int* destorphancount, int* sourceorphancount) {
    auto destcount = 0; 
    auto sourcecount = 0;
    auto readindex = redzonestart + start;
    for (auto i = start; i < end; i++) {
        const auto dest_data = removals[i];
        const auto dest = dest_data & (~deleted);
        const auto isDestValid = (dest < redzonestart);
        const auto isSourceValid = (dest_data >= 0); //GetIsSourceValid(dest);

        const auto isSourceOrphan = (isSourceValid && (!isDestValid));
        const auto isDestOrphan = ((!isSourceValid) && isDestValid);
        
        if (isDestOrphan)[[unlikely]] {
            assert(!is_source_orphan);
            removals[start + destcount++] = dest;
        }
        else if (isSourceOrphan)[[unlikely]] {
            const auto source = list[readindex];
            assert(!is_dest_orphan);
            list[redzonestart + start + sourcecount++] = source;
        }
        else if (isSourceValid && isDestValid)[[likely]] {
            assert(!isSourceOrphan);
            assert(!isDestOrphan);
            const auto source = list[readindex];
            list[dest] = source;
        }
        readindex++;
    } //for i
    
    *sourceorphancount = sourcecount;
    *destorphancount = destcount;
}

struct work_section_t {
    int dest_start_row;
    int dest_start_offset;
    int source_start_row;
    int source_start_offset;
    int count;
};

static constexpr auto out_of_bounds = -1;

//because each thread recalculates its own data, phase 3a and 3b can be combined.
void phase3a_loadbalance(const int* const source_orphan_counts, const int* const dest_orphan_counts, const int thread_count, const int tid, work_section_t& output) {   
    //printf("T: %i, phase 3: thread count = %i\n", tid, thread_count);
    //ascending prefix: 4,1,3 -> 0,4,5,8

    const auto totalcount = dest_orphan_counts[thread_count];
    const auto orphans_per_thread = divup(totalcount, thread_count);
    
    //if we have fewer orphans than threads, declare some out_of_bounds
    const auto startindex = (tid >= totalcount) ? out_of_bounds : tid * orphans_per_thread;
    if (startindex < 0) { output.count = 0; return; }
    
    //find the row to start processing in, remember counts can be 0.
    //for simplicity start the search from 0.
    //replace with a binary search in production 
    auto d = thread_count - 1;
    while (dest_orphan_counts[d] > startindex) { d--; } //<= because rowcount can be 0
    output.dest_start_offset = startindex - dest_orphan_counts[d]; //-1 if out_of_bounds
    output.dest_start_row = d; //do not use for out_of_bounds check

    auto s = thread_count - 1;
    while (source_orphan_counts[s] > startindex) { s--; }
    output.source_start_offset = startindex - source_orphan_counts[s];
    output.source_start_row = s;
    
    //The last thread has fewer items to process, also nthreads can be greater than items
    //output.count = (out_of_bounds == startindex) ? 0 : orphans_per_thread;
    const auto beforecount = tid * orphans_per_thread;
    output.count = std::max(0, std::min(totalcount - beforecount, orphans_per_thread));

    assert(output.count >= 0);
}

__host__ void phase3b_process_orphans(int* list, const int* const sourceorphans, const int* const destorphans, const int* const sourceorphancounts, const int* const destorphancounts, const int tid, const work_section_t& input, const int stride) {
    auto count = input.count;
    if (0 == count) { return; }
    
    auto s = input.source_start_row;
    auto s_index0 = s * stride;
    auto s_offset = input.source_start_offset;
    auto si = sourceorphancounts[s] + s_offset;

    auto d = input.dest_start_row;
    auto d_index0 = d * stride;
    auto d_offset = input.dest_start_offset;
    auto di = destorphancounts[d] + d_offset;
    auto smax = sourceorphancounts[s + 1];
    auto dmax = destorphancounts[d + 1];
    
    while (count--) {
        const auto source = sourceorphans[s_index0 + s_offset++];
        const auto dest   = destorphans  [d_index0 + d_offset++];
        si++; di++;
        list[dest] = source;
        //while, because a row can be empty
        while (si == smax) { s++; s_index0 += stride; s_offset = 0; smax = sourceorphancounts[s + 1]; }
        while (di == dmax) { d++; d_index0 += stride; d_offset = 0; dmax = destorphancounts[d + 1]; }
    }    
}


#define debugprint 1
#undef debugprint

void thread_test() {
    
    constexpr auto length = 205;
    std::vector<int> list(length, 0);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, length), [&](const tbb::blocked_range<size_t>& r){
        for (auto i = r.begin(); i < r.end(); i++) {
            list[i] = i;
        }
    });
    for (auto i = 0; i < length; i++) {
        printf("%i ", list[i]);
    }
    printf("\n");
}

void redzone_CPU(int* list, int listsize, int* removals, int removecount, int(&times)[4]) {
    auto timer = bs::timer(); //auto starts

    const auto threadcount = int(std::thread::hardware_concurrency() - 1);
    const auto stride = divup(removecount, threadcount);
    #ifdef debugprint
        printf("threadcount = %i, removecount = %i\n", threadcount, removecount);
    #endif
    std::vector<int> destorphancounts(threadcount + 1, 0);
    std::vector<int> sourceorphancounts(threadcount + 1, 0);
    #ifdef debugprint
        std::vector<std::array<int, 2>> range(threadcount);
    #endif

    tbb::parallel_for(tbb::blocked_range<size_t>(0, threadcount), [&](const tbb::blocked_range<size_t>& r) {
        const auto tid = int(r.begin());
        const auto start = stride * tid;
        const auto end = std::min(start + stride, removecount);
    #ifdef debugprint
        range[tid][0] = start;
        range[tid][1] = end;
    #endif
    #ifdef debugprint
        sout.println("phase 1: tid = ", tid, ", start = ", start, ", end = ", end, ", s = ", s, ", e = ", e);
    #endif
        phase1_mark_redzone(removals, start, end, listsize - removecount);
    });

    #ifdef debugprint
    for (auto i = 0; i < threadcount; i++) {
        printf("T: %i, start = %i, end= %i, count = %i\n", i, range[i][0], range[i][1], range[i][1] - range[i][0]);
    }
    #endif
    const auto phase1_time = timer.current_us();
    const auto redzonestart = listsize - removecount;

    tbb::parallel_for(tbb::blocked_range<size_t>(0, threadcount), [&](const tbb::blocked_range<size_t>& r) {
        const auto tid = int(r.begin());
        int* dcounts = &destorphancounts[tid];
        int* scounts = &sourceorphancounts[tid];
        const auto start = stride * tid;
        const auto end = std::min(stride * (tid + 1), removecount);
        phase2_delete_and_collect_orphans(list, removals, start, end, redzonestart, dcounts, scounts);
        #ifdef debugprint
            sout.println("T:", tid, ", dcounts[", tid, "] = ", dcounts[tid], ", scounts[", tid, "] = ", scounts[tid]);
        #endif
    });
    #ifdef debugprint
        auto st = 0;
        auto dt = 0;
        for (auto i = 0; i < threadcount; i++) {
            st += sourceorphancounts[i];
            dt += destorphancounts[i];
            printf("T: %i, d = %i(dt=%i), s = %i(st=%i)\n", i, destorphancounts[i], dt, sourceorphancounts[i], st);
        }
    #endif
    const auto phase2_time = timer.current_us();
    auto totalsource = 0;
    auto totaldest = 0;
    for (auto i = 0; i < threadcount; i++) { //recomputes the prefix sum, all threads use the same prefix sum
        const auto d = destorphancounts[i];
        destorphancounts[i] = totaldest;
        totaldest += d;

        const auto s = sourceorphancounts[i];
        sourceorphancounts[i] = totalsource;
        totalsource += s;
    }
    destorphancounts[threadcount] = totaldest;
    sourceorphancounts[threadcount] = totalsource;
    #ifdef debugprint
        for (auto i = 0; i <= threadcount; i++) {
            printf("T: %i, destcount[%i] = %i, sourcecount[%i] = %i\n",i, i, destorphancounts[i], i, sourceorphancounts[i]);
        }
    #endif
   
    std::vector<work_section_t> output(threadcount);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, threadcount), [&](const tbb::blocked_range<size_t>& r) {
        const auto tid = int(r.begin());
        phase3a_loadbalance(&sourceorphancounts[0], &destorphancounts[0], threadcount, tid, output[tid]);
    });
    #ifdef debugprint
        for (auto i = 0; i < threadcount; i++) {
            printf("T:%i, c = %i, dest_start = %i, dest_offset = %i, source_start = %i, source_offset = %i\n", i, output[i].count, output[i].dest_start_offset, output[i].dest_start_row, output[i].source_start_offset, output[i].source_start_row);
        }
    #endif
    tbb::parallel_for(tbb::blocked_range<size_t>(0, threadcount), [&](const tbb::blocked_range<size_t>& r) {
        const auto tid = int(r.begin());
        auto sourceorphans = &list[redzonestart];
        auto destorphans = &removals[0];
        phase3b_process_orphans(list, sourceorphans, destorphans, &sourceorphancounts[0], &destorphancounts[0], tid, output[tid], stride);
    });
    
    const auto phase3_time = timer.current_us();
    times[3] = int(phase3_time - phase2_time);
    times[2] = int(phase2_time - phase1_time);
    times[1] = int(phase1_time);
    times[0] = int(phase3_time);
}

void test_redzone_CPU(const int run) {
    constexpr auto maxelements = 1 << 29;
    TestLists_t TestList(maxelements, maxelements);
    //auto list = TestList.dev_A;
    //auto removals = TestList.dev_R;
    const int rp[] = { -1, 100, 75, 50, 25, 0 };

    int Times[4]; //phase 1..3 + total time
    for (auto redzone_removals = 0; redzone_removals < 1; redzone_removals++) {
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
            for (auto e = 14; e <= 29; e++) {
                const auto listsize = 1ull << e;
                const auto k = (listsize * kpercentage) / 100;
                //const auto greensize = listsize - k;
                //const auto redstart = greensize;

                if (kpercentage > 50 && redzone_percentage != -1) { continue; } //skip redzone investigations for 90%

                TestList.Update(int(listsize), int(k), redzone_percentage);

                //init the data
                const auto StartCPUTime = std::chrono::high_resolution_clock::now();
                redzone_CPU(TestList.host_A, TestList.Asize, TestList.host_R, TestList.Rsize, Times);
                const auto StopCPUTime = std::chrono::high_resolution_clock::now();
                const auto CPUTime = (int)std::chrono::duration_cast<std::chrono::microseconds>(StopCPUTime - StartCPUTime).count();
                
                
                //for short runs this will include the >= 12-25us kernel startup time
                //and will arbitrarily assign this to the phases, but that's fine                
                const auto is_OK = TestList.isOK_CPU();
                //const auto is_OK = true;
                const auto status = is_OK ? "Pass" : "##### error!!!!!";
                printf("n = 1 << %i, k = %i%%, r = %i%%, time1 = %i, time2 = %i, time3 = %i, timeall = %i, cputime = %i, OK = %s, run = %i\n", e, kpercentage, redzone_percentage, Times[1], Times[2], Times[3], Times[0], CPUTime, status, run);
            }
        }
    }
}