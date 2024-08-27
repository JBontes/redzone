constexpr auto GPUBlockCount = 46;
constexpr auto GPUBlockSize = 1024;

#include "DataStruct.hxx"
#include "cuCompactor.cuh"

#include "Thrust_remove.hxx"
#include "redzone.hxx"

#include "Redzone_CPU.hxx"
#include "SDC_paper.hxx"
#include "std_parallel_remove.hxx"
#include <cuda.h>
#include <cuda/pipeline>

__device__ uint64_t klok64() {
    uint64_t clk;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(clk) :: "memory");
    return clk;
}

#define randBound (2)//<100
void initData(int *h_data, int NELEMENTS, int &goodElements, bool randomOrStride) {

    const auto stride = 4;
    for (auto i = 0; i < NELEMENTS; ++i) {
        if (randomOrStride) { h_data[i] = i % stride; }
        else { h_data[i] = (rand() % 100 < randBound) ? 1 : 0; }
        if (h_data[i]) { goodElements++; }
    }
}

void printData(const int *h_data, const int NELEMENTS) {
    for (auto i = 0; i < NELEMENTS; ++i) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;
}

int checkVector(const int *h_data, const int NELEMENTS, const int NgoodElements) {
    //printf("Checking: %i, %i",NELEMENTS,NgoodElements);
    int_predicate predicate;
    auto errorcount1 = 0;
    for (auto i = 0; i < NgoodElements; i++) {
        //assert(predicate(h_data[i]));
        if (!predicate(h_data[i])) { errorcount1++; }
        if (1 == errorcount1) { printf("error at h_data[%i] = %i\n", i, h_data[i]); }
    }
    auto errorcount2 = 0;
    for (auto i = NgoodElements; i < NELEMENTS; i++) {
        //assert(!predicate(h_data[i]));
        if (predicate(h_data[i])) { errorcount2++; }
        if (1 == errorcount2) { printf("error at h_data[%i] = %i\n", i, h_data[i]); }
    }
    return errorcount1 + errorcount2;
}

[[nodiscard]] int* allocGPUMem() {
    int* GPUMem;
    CUDA(cudaMalloc(&GPUMem, 1024 * 1024 * 1024)); //1 GB
    return GPUMem;
}

//get the overhead for the GPU cycle clock. About 2-6 GPU cycles.
__global__ void TimeClock64() {
    volatile const auto StartTime = klok64();
    volatile const auto Time = klok64() - StartTime;
    printf("Time for clock64 = %i\n", int(Time));
}

//The built in cuda memcpy_async has some overhead, because it includes workarounds for invalid input
//I've removed these here to speed things up.
//avoid __isShared + isGlobal + loop overhead in production
//assumes pipeline<cuda::thread_scope_thread>
template <typename T>
__device__ inline void simple_memcpy_async(T* dest, const T* source, const int size) {
    assert(__isGlobal(source));
    assert(__isShared(dest));
    assert((sizeof(T) == size) || (0 == size));
    static_assert((sizeof(T) == 4) || (sizeof(T) == 8));
    if constexpr (sizeof(T) == 4) {
        auto __dest = (uint32_t*)dest;
        auto __source = (uint32_t*)source;

        asm volatile ("cp.async.ca.shared.global [%0], [%1], 4, %2;"
            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(__dest))),
            "l"(__source), "r"(size)
            : "memory");
    }
    if constexpr (sizeof(T) == 8) {
        auto __dest = (uint64_t*)dest;
        auto __source = (uint64_t*)source;
        asm volatile ("cp.async.ca.shared.global [%0], [%1], 8, %2;"
            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(__dest))),
            "l"(__source), "r"(size)
            : "memory");
    }
}


__device__ int globaldata[64 * (1024 + 1)];

__global__ void TimeMemcpyAsync() {
    __shared__ int buffer[32];
    auto pipeline = cuda::make_pipeline();
    for (auto i = threadIdx.x; i < 1024 * 32; i+= blockDim.x) {
        globaldata[i] = threadIdx.x * 10;
    }
    const auto size = (threadIdx.x == 0) ? 4 : 0;
    const auto location = threadIdx.x * 519;
    __syncthreads();

    volatile auto StartTimeAsync = klok64();
    simple_memcpy_async(&buffer[threadIdx.x], &globaldata[threadIdx.x], size);
    pipeline.producer_commit();
    
    volatile auto EndTime = klok64();

    volatile auto StartTimeReg = klok64();
    const auto test = (size) ? globaldata[threadIdx.x] : 0;
    //buffer[threadIdx.x] = test;
    
    volatile auto EndRegTime = klok64();
    __syncthreads();
    
    const auto time = int(EndTime - StartTimeAsync);
    const auto regtime = int(EndRegTime - StartTimeReg);
    
    pipeline.consumer_wait();
    const auto test2 = buffer[threadIdx.x];
    printf("T:%i, Time for memcpy async = %i, data = %i, time for fill regs = %i, data = %i\n", threadIdx.x, time, test2, regtime, test);
}


int main() {
    DisplayHeader();
    printf("test __insert test here__\n");
    constexpr auto OneBlock = true;
    constexpr auto AllBlocks = false;
    for (auto i = 0; i < 20; i++) {
        //uncomment the test you want to run
        //********** CPU tests
        //test_SDC_compact_CPU(i);
        //test_par_remove(i);
        //test_redzone_CPU(i);
        //********** GPU tests NVidia CUDA only
        //Tests on all blocks, remember to adjust line 1 in the top of this file
        //GPUBlockCount = number of SMs in your GPU.
        //test_thrust(i);
        test_Ink(i);
        //test_redzone<AllBlocks>(i);

        //********* GPU tests on on a single block of 1024 threads
        //test_redzone<OneBlock>(i);
        //test_ON_OneBlock(i);
    }
}








