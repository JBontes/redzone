
/*
 * cuCompactor.h
 *
 *  Created on: 21/mag/2015
 *      Author: knotman
 */

#ifndef CUCOMPACTOR_H_
#define CUCOMPACTOR_H_

#include <thrust/scan.h>
#include <cub/cub.cuh> 
#include <thrust/device_vector.h>
#include "Macros.hxx"
#include <chrono>

//namespace cuCompactor {

static constexpr auto FULL_MASK = 0xffffffffu;

    __host__ __device__ constexpr int divup(int n, int d) { return (n + d - 1) / d; }
                                                            
    __host__ __device__ constexpr int pow2i(int e) {
        return 1 << e;
    }


    template <typename T, typename Predicate>
    __global__ void computeBlockCounts(T* d_input, int length, int* d_BlockCounts, Predicate predicate) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        
        int pred = (idx < length) ? predicate(d_input[idx]) : 0;
        int BC = __syncthreads_count(pred);

        if (threadIdx.x == 0) {
            d_BlockCounts[blockIdx.x] = BC; // BC will contain the number of valid elements in all threads of this thread block
        }
    }



    template <int numWarps, typename T, typename Predicate>
    __global__ void compactK(T* d_input, int length, T* d_output, int* d_BlocksOffset, Predicate predicate) {
        const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
        __shared__ int warpTotals[32];
        const auto input = (idx < length) ? d_input[idx] : 0;
        int pred = (idx < length) ? predicate(d_input[idx]) : 0;
        const auto BlockOffset = (pred) ? d_BlocksOffset[blockIdx.x] : 0;
        int w_i = threadIdx.x / warpsize; //warp index
        int w_l = idx % warpsize;//thread index within a warp

        // compute exclusive prefix sum based on predicate validity to get output offset for thread in warp
        int t_m = FULL_MASK >> (warpsize - w_l); //thread mask
        int b = __ballot_sync(FULL_MASK, pred);
        int t_u = __popc(b & t_m); // popc count the number of bit one. simply count the number predicated true BEFORE MY INDEX

        //total valid counts for the warp
        warpTotals[w_i] = __popc(b);
        

        // need all warps in thread block to fill in warpTotals before proceeding
        __syncthreads();

        // first numWarps threads in first warp compute exclusive prefix sum to get output offset for each warp in thread block
        //int numWarps = blockDim.x / warpsize;
        unsigned int numWarpsMask = FULL_MASK >> (warpsize - numWarps);
        if (w_i == 0 && w_l < numWarps) {
            int w_i_u = 0;
            #pragma loop unroll
            for (auto j = 0; j <= 5; j++) { // must include j=5 in loop in case any elements of warpTotals are identically equal to 32
                int b_j = __ballot_sync(numWarpsMask, warpTotals[w_l] & pow2i(j));
                w_i_u += (__popc(b_j & t_m)) << j;
                //printf("indice %i t_m=%i,j=%i,b_j=%i,w_i_u=%i\n",w_l,t_m,j,b_j,w_i_u);
            }
            warpTotals[w_l] = w_i_u;
        }

        // need all warps in thread block to wait until prefix sum is calculated in warpTotals
        __syncthreads();

        // if valid element, place the element in proper destination address based on thread offset in warp, warp offset in block, and block offset in grid
        if (pred) {
            d_output[t_u + warpTotals[w_i] + BlockOffset] = input;
        }
        
    }

    template <class T>
    __global__  void printArray_GPU(T* hd_data, int size, int newline) {
        auto w = 0;
        for (auto i = 0; i < size; i++) {
            if (i % newline == 0) {
                printf("\n%i -> ", w);
                w++;
            }
            printf("%i ", hd_data[i]);
        }
        printf("\n");
    }



    template <int blockSize, int blockCount, typename T, typename Predicate>
    int compact(T* d_input, T* d_output, int length, Predicate predicate, int& time) {
        constexpr auto numWarps = divup(blockSize, 32);
        int numBlocks = divup(length, blockSize);
        if constexpr(blockCount == 1) {
            numBlocks = 1;
        }
        int* d_BlocksCount;
        int* d_BlocksOffset;
        CUDA(cudaMalloc(&d_BlocksCount, sizeof(int) * numBlocks));
        CUDA(cudaMalloc(&d_BlocksOffset, sizeof(int) * numBlocks));
        thrust::device_ptr<int> thrustPrt_bCount(d_BlocksCount);
        thrust::device_ptr<int> thrustPrt_bOffset(d_BlocksOffset);
        const auto start = std::chrono::high_resolution_clock::now();
        
        
        //phase 1: count number of valid elements in each thread block
        computeBlockCounts <<<numBlocks, blockSize >>> (d_input, length, d_BlocksCount, predicate);
        CUDA(cudaDeviceSynchronize());
        //phase 2: compute exclusive prefix sum of valid block counts to get output offset for each thread block in grid
        //we only have a single block
        //if constexpr (blockCount > 1) {
            thrust::exclusive_scan(thrustPrt_bCount, thrustPrt_bCount + numBlocks, thrustPrt_bOffset);
        //}
        

        //phase 3: compute output offset for each thread in warp and each warp in thread block, then output valid elements
        compactK<32> <<<numBlocks, blockSize, sizeof(int)* (blockSize / warpsize)>>> (d_input, length, d_output, d_BlocksOffset, predicate);
        CUDA(cudaDeviceSynchronize());
        // determine number of elements in the compacted list
        const auto end = std::chrono::high_resolution_clock::now();
        time = int(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
        int compact_length = thrustPrt_bOffset[numBlocks - 1] + thrustPrt_bCount[numBlocks - 1];
        
        CUDA(cudaFree(d_BlocksCount));
        CUDA(cudaFree(d_BlocksOffset));

        return compact_length;
    }

    struct int_predicate {
        __host__ __device__
            bool operator()(const int x) {
            return (x >= 0);
        }
    };

    void test_Ink(const int run) {
        constexpr auto maxelements = 1 << 29;
        TestLists_t TestList(maxelements, maxelements);
    //srand(time(0));
        auto errors = 0;
        int kp[] = { 2, 10, 50, 90 };
        //data elements from 2^5 to 2^29
        for (auto r = 0; r < 4; r++) {
            for (auto e = 29; e >= 14; e--) {
                const auto NELEMENTS = 1 << e;
                constexpr auto blockSize = GPUBlockSize;
                const auto k = (uint64_t(NELEMENTS) * kp[r]) / 100;
                TestList.Update(NELEMENTS, k);
                //const auto NgoodElements = NELEMENTS - k;
                const auto datasize = sizeof(int) * NELEMENTS;
                //host input/output data
                auto d_data = TestList.dev_A;
                auto d_output = TestList.dev_A2;
                auto h_data = TestList.host_A;
                const auto start1 = std::chrono::high_resolution_clock::now();
                
                remove_phase1 <<<GPUBlockCount, blockSize>>>(d_data, TestList.dev_R, k);
                
                CUDA(cudaDeviceSynchronize());
                const auto end1 = std::chrono::high_resolution_clock::now();
                int time2;
                const auto NewLength = compact<GPUBlockSize, GPUBlockCount>(d_data, d_output, NELEMENTS, int_predicate(), time2);
                const auto t1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count();

                //copy back results to host
                CUDA(cudaMemcpy(h_data, d_output, datasize, cudaMemcpyDeviceToHost));
                //printData(h_data,NELEMENTS);
                const auto is_OK = TestList.isOK();
                errors += (!is_OK);
                printf("n = %i, k = %i%%, t1 = %i, t2 = %i, %s, run = %i\n", e, kp[r], int(t1), time2, is_OK ? "Pass" : "Fail", run);
            } //for blocksize
        } //for elements
    }

//} ///* namespace cuCompactor */
#endif ///* CUCOMPACTOR_H_ */