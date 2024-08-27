/*
* 
@article{10.1155/2018/2037272,
author = {Bernab\'{e}, Gregorio and Acacio, Manuel E. and K\"{o}stler, Harald},
title = {{On the Parallelization of Stream Compaction on a Low-Cost SDC Cluster}},
year = {2018},
issue_date = {2018},
publisher = {Hindawi Limited},
address = {London, GBR},
volume = {2018},
issn = {1058-9244},
url = {https://doi.org/10.1155/2018/2037272},
doi = {10.1155/2018/2037272},
abstract = {Many highly parallel algorithms usually generate large volumes of data containing both 
valid and invalid elements, and high-performance solutions to the stream compaction problem reveal 
extremely important in such scenarios. Although parallel stream compaction has been extensively studied 
in GPU-based platforms, and more recently, in the Intel Xeon Phi platform, 
no study has considered yet its parallelization using a low-cost computing cluster, 
even when general-purpose single-board computing devices are gaining popularity among 
the scientific community due to their high performance per $ and watt. 
In this work, we consider the case of an extremely low-cost cluster composed by four Odroid C2 
single-board computers (SDCs), showing that stream compaction can also benefit
—important speedups can be obtained—
from this kind of platforms. 
To do so, we derive two parallel implementations for the stream compaction problem using MPI. 
Then, we evaluate them considering varying number of processes and/or SDCs, as well as different input sizes. 
In general, we see that unless the number of elements in the stream is too small, 
the best results are obtained when eight MPI processes are distributed among the four 
SDCs that conform the cluster. To add value to the obtained results, 
we also consider the execution of the two parallel implementations for the stream compaction 
problem on a very high-performance but power-hungry 18-core Intel Xeon E5-2695 v4 multicore processor, 
obtaining that the Odroid C2 SDC cluster constitutes a much more efficient alternative 
when both resulting execution time and required energy are taken into account. 
Finally, we also implement and evaluate a parallel version of the stream split problem 
to store also the invalid elements after the valid ones. 
Our implementation shows good scalability on the Odroid C2 SDC cluster and more compensated 
computation/communication ratio when compared to the stream compaction problem.},
journal = {Sci. Program.},
month = {jan},
numpages = {10}
}
* 

*/


#include <vector>
#include <cassert>
#include <thread>
#include "DataStruct.hxx"

void sequential_stream_compact(const std::vector<int>& input, std::vector<int>& output, int& nvalid) {
    auto k = 0;
    for (auto i = size_t(0); i < input.size(); i++) {
        const auto in = input[i];
        if (in) { output[k++] = in; }
    }
    nvalid = k;
}

void SDC_stream_compact(const int* const Input, const int n, int* Output, int deleteMe, int& nvalid) {
//Input: Vector Input of length n
//Input : Predicate function F
//Input : Number of processes p
//Input : pid of process
//Output : Vector Output of valid elements
//Output : nvalid: the number of valid elements
//(1)  nvalid = 0
    nvalid = 0;
//(2)  stride = n / p
    const auto threadcount = int(std::thread::hardware_concurrency() - 1);
    const auto p = threadcount;
    const auto stride = int(divup(n, p));
    //const auto tail = n - (stride * (p - 1));
//(3)  prefix_sum[0:(stride - 1)] =  0
    auto prefix_sum = std::vector<std::vector<int>>(p);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, threadcount), [&](const tbb::blocked_range<size_t>& r) {
        const auto pid = int(r.begin());
        prefix_sum[pid] = std::vector<int>(stride + 1, 0);
    });
//(4)  V[0:(p - 1)] =  0
    auto V = std::vector<int>(p + 1, 0);
//(5)  for i = 0 to stride - 1 in parallel do
    tbb::parallel_for(tbb::blocked_range<size_t>(0, threadcount), [&](const tbb::blocked_range<size_t>& r) {
        const auto pid = int(r.begin());
        const auto len = std::max(0, std::min(int(n) - (stride * pid), stride));
        const auto start = pid * stride;
//(6)      if F(Input[i]) then
        for (auto i = 0; i < len; i++) {
            if (Input[start + i] != deleteMe) {
//(7)          prefix_sum[i] = 1
//(8)          V[pid]++
                prefix_sum[pid][i] = 1;
                V[pid + 1]++;
//(9)      end if
            }
//(10) end for
        }
        //pr.println("T: ", pid ,", total items: ", len, ", keep: ", V[pid + 1], ", delete: ", len - V[pid + 1]);
    });
//(11) if pid > 0 then
//(12)     Send V[pid] to process pid 0
//(13) end if
//(14) if pid == 0 then
//(15)     for i = 1 to npid do
    for (auto i = 1; i <= p; i++) {
//(16)         Receive V[i]
//(17)         V[i] = V[i] + V[i - 1]
        V[i] += V[i - 1];
        //pr.print_(V[i], " ");
//(18)     end for
    }
    //pr.println();
//(19)     for i = 1 to npid do
//(20)         Send V[i - 1] to process pid i
//(21)     end for
//(22)     nvalid = V[p - 1]
    nvalid = V[p];
//(23) end if
//(24) if pid > 0 then
//(25)     Receive V[pid - 1]
//(26) end if
////(27) prefix_sum[0] = prefix_sum[0] + V[pid - 1]
//        const auto len = std::max(0, std::min(int(n) - (stride * pid), stride));
//        prefix_sum[pid][0] += V[pid];
////(28) for i = 0 to stride - 1 in parallel do
//        for (auto i = 1; i <= len; i++) {
////(29)     prefix_sum[i] = prefix_sum[i - 1] + prefix_sum[i]
//            prefix_sum[pid][i] += prefix_sum[pid][i - 1];        
////(30) end for
//        }
//    });

    //The lack of load balancing will make this perform poorly if the distribution of keeps is skewed.
    tbb::parallel_for(tbb::blocked_range<size_t>(0, threadcount), [&](const tbb::blocked_range<size_t>& r) {
        const auto pid = int(r.begin());
//(31) for i = 0 to stride - 1 in parallel do
        auto readstart = (pid * stride);
        auto writepos = V[pid];
        const auto len = std::max(0, std::min(int(n) - (stride * pid), stride));
        for (auto i = 0; i < len; i++) {
            if (prefix_sum[pid][i]) {
                const auto a = Input[readstart + i];
                Output[writepos++] = a;
                //if (-1 == a) { pr.println("error: x = ", readstart, " + ", i, ", pre= ", Input[readstart+i-1], ", post= ", Input[readstart+i+1], ", writepos= ", writepos-1); } 
            }
//(32)     if prefix_sum[i] != prefix_sum[i - 1] then
//(33)         Output[prefix_sum[i - 1]] = Input[i]
            //if (prefix_sum[pid][i + 1] != prefix_sum[pid][i]) { Output[prefix_sum[pid][i]] = Input[start + i]; }
//(34)     end if
//(35) end for
        }
    });
}

void test_SDC_compact_CPU(const int run) {
    constexpr auto maxelements = 1 << 29;
    TestLists_t TestList(maxelements, maxelements);
    //auto list = TestList.dev_A;
    //auto removals = TestList.dev_R;
    const int rp[] = { -1, 100, 75, 50, 25, 0 };

    int* output = (int*)malloc(sizeof(int) * maxelements);
    assert(nullptr != output);
    int Times[4];

    //CUDA(cudaMalloc(&BlockStore, sizeof(BlockStorage_t<GPUBlocks>)));
    //printf("Test of redzone GPU wide\n");
    //64 KiB = 16K x 4 -> 2^14 x 4;
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

                //**printf("old n = %i, new n = %i, old k = %i, new k = %i\n", TestList.Asize, listsize, TestList.Rsize, k);
                //TestList.SetNK(listsize, k);


                //init the data
                //const auto start1 = std::chrono::high_resolution_clock::now();
                //remove_phase1<<<GPUBlockCount, GPUBlockSize>>>(list, removals, k);

                //printf("MaxAllowedBlocks = %i\n", MaxAllowedBlocks);
                const auto StartCPUTime = std::chrono::high_resolution_clock::now();
                
                auto nvalid = 0;
                SDC_stream_compact(TestList.host_A, TestList.Asize, output, -1, nvalid);
                const auto StopCPUTime = std::chrono::high_resolution_clock::now();
                const auto CPUTime = (int)std::chrono::duration_cast<std::chrono::microseconds>(StopCPUTime - StartCPUTime).count();

                //todo: see if output and input can overlap, but later
                std::memcpy(TestList.host_A, output, nvalid * sizeof(int));
                //for short runs this will include the >= 12-25us kernel startup time
                //and will arbitrarily assign this to the phases, but that's fine

                const auto is_OK = TestList.isOK_CPU();
                const auto status = is_OK ? "Pass" : "##### error!!!!!";
                printf("n = 1 << %i, k = %i%%, r = %i%%, time1 = %i, time2 = %i, time3 = %i, timeall = %i, cputime = %i, OK = %s, run = %i\n", e, kpercentage, redzone_percentage, Times[1], Times[2], Times[3], Times[0], CPUTime, status, run);
            }
        }
    }
}
