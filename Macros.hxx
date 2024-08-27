#ifndef _MACRO_H
#define _MACRO_H
#include <bit>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <cassert>
#include <cstdarg>
#include <chrono>
//#include "intellisense_cuda_intrinsics.h"

#ifdef __INTELLISENSE__
    #define __CUDACC__
    #define __CUDA_ARCH__
    //disable intellisense warnings for cuda functions
    __device__ void __syncthreads() {};
    template <typename T>
    __device__ bool [[nodiscard]] __isShared(T* ptr) {};
    __device__ uint32_t [[nodiscard]] __ballot_sync(uint32_t mask, uint32_t predicate) {};
    __device__ uint64_t [[nodiscard]] clock64() {};
    __device__ uint32_t [[nodiscard]] __ffs(const uint32_t value) {};
    __device__ uint32_t [[nodiscard]] __clz(const uint32_t mask) {};
#endif

#ifndef NDEBUG
#ifndef _DEBUG
#define _DEBUG
#endif
#endif   

//adapted from: https://github.com/bshoshany/thread-pool/blob/master/include/BS_thread_pool_utils.hpp
namespace bs {
    class [[nodiscard]] timer {
    public:
        timer() = default;

        [[nodiscard]] auto current_ms() const {
            return (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time)).count();
        }

        void start() {
            start_time = std::chrono::steady_clock::now();
        }

        void stop() {
            elapsed_time = std::chrono::steady_clock::now() - start_time;
        }

        [[nodiscard]] auto ms() const {
            return (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time)).count();
        }

        [[nodiscard]] auto current_us() const {
            return (std::chrono::duration_cast<std::chrono::microseconds>(elapsed_time)).count();
        }

    private:
        std::chrono::time_point<std::chrono::steady_clock> start_time = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_time = std::chrono::duration<double>::zero();
    }; // class timer
}


#define restrict __restrict__  //https://stackoverflow.com/questions/145270/c-correction-expected-unqualified-id-before-const

using uint32_t = unsigned int;




#ifdef __CUDA_ARCH__
static const int warpsize = 32;  //make warpsize a compile time constant, instead of an intrinsic 
#else
static const int warpsize = 1;
#endif
const int WarpMax = warpsize - 1;
const uint32_t Everyone = uint32_t(-1);

//Threadid within the current block
__host__  __device__ __forceinline__ int ThreadId() {
    #ifdef __CUDA_ARCH__
    assert(threadIdx.y == 0);
    assert(threadIdx.z == 0);
    return threadIdx.x;
    #else
    return 0;  //assume single threaded
    #endif
}

template <uint32_t d>
__host__ __device__ constexpr uint32_t mod(uint32_t n) {
    return n % d;
}

template <uint32_t d>
__host__ __device__ constexpr uint32_t div(uint32_t n) {
    return n / d;
}

//The number of warp in the current block
__host__ __device__ __forceinline__ int WarpCount() {
    #ifdef __CUDA_ARCH__
    return ((blockDim.x + (warpsize - 1)) / warpsize);
    #else
    return 1;
    #endif
}

template <typename T>
T minmax0(const T a, const T b) {
    if constexpr (sizeof(T) == 4) {
        #if __CUDA_ARCH__ >= 900
        return __vimin_s32_relu(a, b);
        #else
        return std::max(std::min(a, b), 0);
        #endif
    } else {
        return std::max(std::min(a, b), 0);
    }
}


template <int Divisor>
constexpr int [[nodiscard]] DivUp(const int Value) {
    return (Value + Divisor - 1) / Divisor;
}

__host__ __device__ constexpr auto [[nodiscard]] divup(std::integral auto n, std::integral auto d) {
    return (n + d - 1) / d;
}

//A sequential warpid with the block
__host__ __device__ __forceinline__ int WarpId() {
    #ifdef __CUDA_ARCH__
    return threadIdx.x / warpsize;
    #else
    return 0;
    #endif
}

__host__ __device__ __forceinline__ int BlockSize() {
    #ifdef __CUDA_ARCH__
    return blockDim.x;
    #else
    return 1;
    #endif
}

template <int LaneCount>
__host__ __device__ uint32_t LaneMask() {
    return (1u << LaneCount) - 1;
}

__host__ __device__ __forceinline__ int GridSize() {
    #ifdef __CUDA_ARCH__
    return gridDim.x * blockDim.x;
    #else
    return 1;
    #endif
}

__host__ __device__ __forceinline__ int DeviceWarpId() {
    #ifdef __CUDA_ARCH__
    const auto result = ((blockIdx.x * blockDim.x) + threadIdx.x) / warpsize;
    //print("((blockIdx: %i * BlockDim: %i = %i) + (threadIdx: %i)) = %i / warpsize: %i) => %i", blockIdx.x, blockDim.x, (blockIdx.x * blockDim.x), threadIdx.x, ((blockIdx.x * blockDim.x) + threadIdx.x), warpsize, result);
    return result;
    #else
    return 0;
    #endif
}

template <typename T>
__host__ __device__ void DebugAtomicInc(T* Address) {
    #ifdef _DEBUG
    #ifdef __CUDA_ARCH__
    atomicAdd(Address, 1);
    #else
    * Address++;
    #endif
    #endif
}

__host__ __device__ int DeviceBlockId() {
    #ifdef __CUDA_ARCH__
    return blockIdx.x;
    #else
    return 0;
    #endif
}

__host__ __device__ int DeviceThreadId() {
    #ifdef __CUDA_ARCH__
    const auto result = (blockIdx.x * blockDim.x) + threadIdx.x;
    return result;
    #else
    return 0;
    #endif
}

__host__ __device__ bool isShared(void* ptr) {
#ifdef __CUDA_ARCH__
    return __isShared(ptr);
#else
    return false;
#endif
}

__host__ __device__ int DeviceThreadCount() {
#ifdef __CUDA_ARCH__
    return gridDim.x * blockDim.x;
#else
    return 1;
#endif
}

__host__ __device__ int BlockId() {
#ifdef __CUDA_ARCH__
    return blockIdx.x;
#else
    return 0;
#endif
}

template <typename T>
__host__ __device__ void memcopy(T* dest, const T* source, size_t length) {
#ifdef __CUDA_ARCH__
    assert((length % sizeof(T)) == 0);
    const auto count = length / sizeof(T);
    for (auto i = threadIdx.x; i < count; i += blockDim.x) {
        dest[i] = source[i];
    }
#else
    memcpy(dest, source, length);
#endif
}

template <typename T>
__host__ __device__ void copy(T* dest, const T* source) {
    constexpr auto length = sizeof(T);
    static_assert((sizeof(T) % sizeof(uint64_t)) == 0);
    const auto dest2 = (uint64_t*)dest;
    const auto source2 = (uint64_t*)source;
    memcopy(dest2, source2, length);
}

template <int ThreadCount>
__device__ constexpr uint32_t ActiveMask() {
    static_assert(ThreadCount <= 32, "ThreadCount must be <= 32");
    static_assert(ThreadCount >= 1, "ThreadCount must be >= 1");
    if constexpr (ThreadCount == 32) {  //suppress warning
        return uint32_t(-1);
    } else {
        const auto result = (1u << ThreadCount) - 1;
        static_assert(std::popcount(result) == ThreadCount);
        return result;
    }
}

#ifdef __CUDA_ARCH__
__device__ uint32_t ActiveMask() {
    return __ballot_sync(Everyone, 1);
}
#endif

#define unreachable() do { __builtin_unreachable(); } while (0)


#ifdef NOT_DEBUG
#define assert(p) do {} while () // do nothing
#endif

//#define NOPRINT
#ifdef NOPRINT
#define printn(format, ...) do {} while (0)  //do nothing
#define print(format, ...) do {} while (0)  //do nothing
#define printonce(format, ...) do {} while (0) //do nothing
#define printwarp(format, ...) do {} while (0) //do nothing
#define printif(predicate, format, ...) do {} while (0) //do nothing
#define printonceif(predicate, format, ...) do {} while (0) //do nothing
#define printwarpif(predicate, format, ...) do {} while (0) //do nothing
#define alwaysprintif(predicate, format, ...) do { } while (0)
#define alwaysprint(format, ...) do { print(format, __VA_ARGS__); } while (0)
#define alwaysprintonce(format, ...) do { print(format, __VA_ARGS__); } while (0)
#define alwaysprintwarp(format, ...) do { print(format, __VA_ARGS__); } while (0)
#else
#ifdef __CUDA_ARCH__
#define print(format, ...) do { printf("c T:%02i W:%02i B:%02i Line:%4i " format "", ThreadId(), WarpId(), blockIdx.x, __LINE__, __VA_ARGS__); } while (0)

#define alwaysprintif(predicate, format, ...) do { if(predicate) { print(format, __VA_ARGS__); } } while (0)
#define alwaysprint(format, ...) do { print(format, __VA_ARGS__); } while (0)
#define alwaysprintonce(format, ...) if (ThreadId() == 0) do { print(format, __VA_ARGS__); } while(0) 
#define alwaysprintwarp(format, ...) if (LaneId() == 0) do { print(format, __VA_ARGS__); } while(0) 
#define alwaysprintwarpif(predicate, format, ...) if (LaneId() == 0 && predicate) do { print(format, __VA_ARGS__); } while(0) 
#define printn(format, ...) do { print(format "\n", __VA_ARGS__); } while (0)
#define printonce(format, ...) do { if (ThreadId() == 0) { print(format, __VA_ARGS__); } } while(0)
#define printwarp(format, ...) do { if (LaneId() == 0) { print(format, __VA_ARGS__); } } while(0)
#define printif(predicate, format, ...) do { if(predicate) { print(format, __VA_ARGS__); } } while(0)
#define printonceif(predicate, format, ...) do { if(predicate && ThreadId() == 0) { print(format, __VA_ARGS__); } } while(0)
#define printwarpif(predicate, format, ...) do { if(predicate && LaneId() == 0) { print(format, __VA_ARGS__); } } while(0)

#else
#ifdef _DEBUG
#define print(format, ...) do { printf("c CPU Line:%4i " format "\n", __LINE__, __VA_ARGS__); } while (0)
#define printn(format, ...) do { print(format, __VA_ARGS__); } while (0)
#define alwaysprintif(predicate, format, ...) do { if(predicate) { print(format, __VA_ARGS__); } } while (0)
#define alwaysprint(format, ...) do { print(format, __VA_ARGS__); } while (0)
#define alwaysprintonce(format, ...) do { print(format, __VA_ARGS__); } while (0)
#define alwaysprintwarp(format, ...) do { print(format, __VA_ARGS__); } while (0)
#define printonce(format, ...) do { if (ThreadId() == 0) { print(format, __VA_ARGS__); } } while(0)
#define printwarp(format, ...) do { if (LaneId() == 0) { print(format, __VA_ARGS__); } } while(0)
#define printif(predicate, format, ...) do { if(predicate) { print(format, __VA_ARGS__); } } while(0)
#define printonceif(predicate, format, ...) do { if((predicate) && ThreadId() == 0) { print(format, __VA_ARGS__); } } while(0)
#define printwarpif(predicate, format, ...) do { if(predicate && LaneId() == 0) { print(format, __VA_ARGS__); } } while(0)
#else
#define print(format, ...) do { fprintf(stderr, "c CPU Line:%4i " format "", __LINE__, __VA_ARGS__); } while (0)
#define printn(format, ...) do { print(format, __VA_ARGS__); } while (0)  
#define printonce(format, ...) do { if (ThreadId() == 0) { print(format, __VA_ARGS__); } } while(0)
#define printwarp(format, ...) do { if (LaneId() == 0) { print(format, __VA_ARGS__); } } while(0)
#define printif(predicate, format, ...) do { if(predicate) { print(format, __VA_ARGS__); } } while(0)
#define printonceif(predicate, format, ...) do { if constexpr(predicate) { if(ThreadId() == 0) { print(format, __VA_ARGS__); }}} while(0)
#define printwarpif(predicate, format, ...) do { if(predicate && LaneId() == 0) { print(format, __VA_ARGS__); } } while(0)
#define alwaysprintif(predicate, format, ...) do { if(predicate) { print(format, __VA_ARGS__); } } while (0)
#define alwaysprint(format, ...) do { print(format, __VA_ARGS__); } while (0)
#define alwaysprintonce(format, ...) do { print(format, __VA_ARGS__); } while (0)
#define alwaysprintwarp(format, ...) do { print(format, __VA_ARGS__); } while (0)
#endif
#endif
#endif


#define alwaysassert(test) { if(!(test)) { printf("c T:%02i W:%02i B:%02i assert (%s) failed in line %i and file %s\n", ThreadId(), WarpId(), BlockId(), #test, __LINE__, __FILE__); }} while (0)

#ifdef _DEBUG
#ifdef __CUDA_ARCH__
#define assert1(test) do { if(!(test)) { printf("B:%i T:%i: assert failed in line %i and file %s\n", blockIdx.x, ThreadId(), __LINE__, __FILE__); assert(false); }} while (0)
#define assert2(test, format, ...)  do { if(!(test)) { printf("B:%i, T:%i: assert failed in line %i and file %s\n" format "\n", blockIdx.x, ThreadId(), __LINE__, __FILE__, __VA_ARGS__); assert(false); }} while (0)
#define assertAllEqual(test) do { if (__ballot_sync(__activemask(), 1) != -1 || __ballot_sync(Everyone, __shfl_sync(Everyone, (int)(test), 0) == (int)(test)) != -1 ) { print("assert failed: activemask = $%x, value: %i not equal across the warp", __ballot_sync(__activemask(), 1), (int)(test)); } } while (0)
#else
#define assert1(test) do { if(!(test)) { printf("CPU: assert failed in line %i and file %s\n", __LINE__, __FILE__); }} while (0)
#define assert2(test, format, ...) do { if(!(test)) { printf("CPU: assert failed in line %i and file %s\n" format "\n", __LINE__, __FILE__, __VA_ARGS__);  }} while (0)
#define assertAllEqual(test) do {} while (0)
#endif
#else
#define assert1(test) do {} while (0)
#define assert2(test, format, ...) do {} while (0)
#define assertAllEqual(test) do {} while (0)
#endif    

#ifndef NOCUDA
__host__ __device__ inline void cudaassert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (cudaSuccess != code) {
        #ifdef __CUDA_ARCH__ 
        printf("GPUassert: code = %i %s %d\n", code, file, line);
        if (abort) { assert1(false); }
        #else
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) { exit(code); }
        #endif
    }
}

#define CUDA(statuscode) cudaassert((statuscode), __FILE__, __LINE__)
#endif

//prevent a = b errors when a == b was intended.
//also prevents narrowing conversions of either a or b
template <typename T>
__host__ __device__ bool isEqual(const T a, const T b) {
    return a == b;
}

__host__ __device__ bool isAssigned(void* ptr) {
    return (nullptr != ptr);
}


//https://stackoverflow.com/questions/44337309
__host__ __device__ __forceinline__ uint32_t LaneId() {
    uint32_t result;
#ifdef __CUDA_ARCH__
    //asm ("mov.u32 %0, %%laneid;" : "=r"(result)); //on newer GPUs the special registers are faster
    result = mod<32>(ThreadId());
#else
    result = 0;
#endif
    return result;
}

//diagnose synchronization errors if needed
__host__ __device__ void syncwarp(uint32_t mask = uint32_t(-1), const int line = 0) {
    #if __CUDA_ARCH__ >= 700
        //if (line != 0) { 
        //    if (LaneId() == 0) { printf("w%i, t%i, l%i before syncwarp()\n", WarpId(), ThreadId(), line); } 
        //    //__syncwarp(mask);
        //    //if (LaneId() == 0) { printf("w%i, t%i, l%i after syncwarp()\n", WarpId(), ThreadId(), line); } 
        //}
    __syncwarp(mask);
    #endif
}

__host__ __device__ __forceinline__ void syncthreads(const int line = 0) {
    #ifdef __CUDA_ARCH__
        //if (line != 0) { 
        //    if (LaneId() == 0) { printf("w%i t%i, l%i before syncthreads\n", WarpId(), ThreadId(), line); } 
        //    //__syncthreads();
        //    //if (LaneId() == 0) { printf("w%i t%i, l%i after syncthreads\n", WarpId(), ThreadId(), line); } 
        //}
    __syncthreads();
    #endif
}


enum CachePolicy_t { cp_L2only, cp_L1_L2 };

//avoid the loop and __isShared + isGlobal overhead in production
#ifdef __CUDA_ARCH__
template <typename T>
__device__ inline void memcpy_async(T* dest, const T* source, const int size) {
    assert(__isGlobal(source));
    assert(__isShared(dest));
    assert((sizeof(T) == size) || (0 == size));
    static_assert((sizeof(T) == 4) || (sizeof(T) == 8));
    if constexpr (sizeof(T) == 4) {
        auto __dest = (uint32_t*)dest;
        auto __source = (uint32_t*)source;
        //.ca = L1+L2, .cg = only L2 cache
        //.cg can only work for transfers of 16 bytes per thread
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
#endif
enum threadprint { all_, once_, warp_ };

template <bool DoPrint = true, threadprint ThreadPrint = all_>
__host__ __device__ void displayif(const bool predicate, const char* fmt, ...) {
    if constexpr (DoPrint) {
        auto DoPrint = predicate;
        if constexpr (ThreadPrint == once_) {
            DoPrint = (ThreadId() == 0);
        } else if constexpr (ThreadPrint == warp_) {
            DoPrint = (LaneId() == 0);
        } else if constexpr (ThreadPrint == all_) {
            DoPrint = true;
        } else {
            static_assert((ThreadPrint >= all_) && (ThreadPrint <= warp_));
        }
        if (DoPrint) {
            va_list args;
            va_start(args, fmt);
            printf("c T:%02i W:%02i B:%02i Line:%4i ", ThreadId(), WarpId(), blockIdx.x, __LINE__);
            printf(fmt, args);
        }
    }
}

template <bool DoPrint = true, threadprint ThreadPrint = all_>
__host__ __device__ void display(const char* fmt, ...) {
    if constexpr (DoPrint) {
        va_list args;
        va_start(args, fmt);
        displayif<DoPrint, ThreadPrint>(true, fmt, args);
    }
}

#ifdef __CUDA_ARCH__
__device__ float GlobalTimeToMillisecs(const uint64_t ElapsedGlobalTime) {
    //A predefined, 64 - bit global nanosecond timer.
    return ElapsedGlobalTime / 1'000'000;
}

__device__ float GlobalTimeToMicroSecs(const uint64_t ElapsedGlobalTime) {
    //A predefined, 64 - bit global nanosecond timer.
    return ElapsedGlobalTime / 1'000;
}

//clock64 counts cycles, not nanoseconds
__device__ float MillisecondsSinceStart(const uint64_t Start) {
    return GlobalTimeToMillisecs(clock64() - Start);
}

__device__ float MicrosecondsSinceStart(const uint64_t Start) {
    return GlobalTimeToMicroSecs(clock64() - Start);
}
#endif

//A predefined, 32 - bit mask with a bit set in the position equal to the thread's lane number in the warp. 
__host__ __device__ __forceinline__ uint32_t lanemask_eq() {
    uint32_t result;
#ifdef __CUDA_ARCH__
    //asm ("mov.u32 %0, %%lanemask_eq;" : "=r"(result));
    result = 1u << LaneId();
#else
    result = 1;
#endif
    return result;
}

//32-bit mask with bits set in positions less than the thread's lane number in the warp. 
__host__ __device__ __forceinline__ uint32_t lanemask_lt() {
    uint32_t result;
#ifdef __CUDA_ARCH__
    result = -1u >> (32 - LaneId());
#ifdef _DEBUG
    uint32_t result2;
    asm ("mov.u32 %0, %%lanemask_lt;" : "=r"(result2));
    if (result2 != result) { print("result2: %u != result: %u\n", result2, result); }
#endif
#else
    result = 0;
#endif
    return result;
}

//32 - bit mask with bits set in positions less than or equal to the thread's lane number in the warp. 
__host__ __device__ __forceinline__ uint32_t lanemask_le() {
    uint32_t result;
#ifdef __CUDA_ARCH__
    result = -1u >> (31 - LaneId());
#ifdef _DEBUG
    uint32_t result2;
    asm ("mov.u32 %0, %%lanemask_le;" : "=r"(result2));
    if (result2 != result) { print("result2: %u != result: %u\n", result2, result); }
#endif
#else
    result = 1;
#endif
    return result;
}

//32-bit mask with bits set in positions greater than or equal to the thread's lane number in the warp. 
__host__ __device__ __forceinline__ uint32_t lanemask_ge() {
    uint32_t result;
#ifdef __CUDA_ARCH__
    result = -1u << (LaneId());
#ifdef _DEBUG
    uint32_t result2;
    asm ("mov.u32 %0, %%lanemask_ge;" : "=r"(result2));
    if (result2 != result) { print("result2: %u != result: %u\n", result2, result); }
#endif
#else
    result = 1;
#endif
    return result;
}

//32-bit mask with bits set in positions greater than the thread's lane number in the warp. 
__host__ __device__ __forceinline__ uint32_t lanemask_gt() {
    uint32_t result;
#ifdef __CUDA_ARCH__
    result = -1u << (LaneId() + 1);
#ifdef _DEBUG
    uint32_t result2;
    asm volatile("mov.u32 %0, %%lanemask_gt;" : "=r"(result2));
    if (result2 != result) { print("result2: %u != result: %u\n", result2, result); }
#endif
#else
    result = 1;
#endif
    return result;
}

//#ifdef __CUDA_ARCH__
__device__ [[nodiscard]] uint32_t GetLeader(const int BallotMask) {
    return __ffs(BallotMask) - 1;
}

__device__ bool [[nodiscard]] isLeader(const uint32_t BallotMask) {
    return GetLeader(BallotMask) == LaneId();
}

__device__ [[nodiscard]] bool isLeader(const uint32_t BallotMask, const uint32_t laneid) {
    assert(laneid < 32);
    return GetLeader(BallotMask) == laneid;
}


enum where { lt, le, eq, neq, gt, ge, biggest, smallest, nle, nlt, next };
template <where Where = where::lt>
__device__ [[nodiscard]] int LanesBefore(const int BallotMask) {
    switch (Where) {
        case lt:       return __popc(BallotMask & lanemask_lt());
        case le:       return __popc(BallotMask & lanemask_le());
        case eq:       return __popc(BallotMask & lanemask_eq());
        case neq:      return __popc((~BallotMask) & lanemask_eq());
        case gt:       return __popc(BallotMask & lanemask_gt());
        case ge:       return __popc(BallotMask & lanemask_ge());
        case nle:      return __popc((~BallotMask) & lanemask_le());
        case nlt:      return __popc((~BallotMask) & lanemask_lt());
        case biggest:  return 31 - __clz(BallotMask);
        case smallest: return __ffs(BallotMask) - 1;
        case next:     return __ffs(BallotMask & lanemask_ge()) - 1;
        default: assert1(false); return 0;
    }
}

//https://stackoverflow.com/a/5689133/650492
void DisplayHeader() {
    const int kb = 1024;
    const int mb = kb * kb;
    std::cout << "Redzone Thrust stats" << std::endl << "=========" << std::endl << std::endl;

    std::cout << "CUDA version:   v" << CUDART_VERSION << std::endl;
    #ifdef THRUST_MAJOR_VERSION 
    std::cout << "Thrust version: v" << THRUST_MAJOR_VERSION << "." << THRUST_MINOR_VERSION << std::endl << std::endl;
    #endif

    int devCount;
    CUDA(cudaGetDeviceCount(&devCount));
    std::cout << "CUDA Devices: " << std::endl << std::endl;

    for (auto i = 0; i < devCount; ++i) {
        cudaDeviceProp props;
        CUDA(cudaGetDeviceProperties(&props, i));
        std::cout << i << ": " << props.name << ": " << props.major << "." << props.minor << std::endl;
        std::cout << "  Clock speed              " << props.clockRate / 1000000.0f << "Ghz\n";
        std::cout << "  Global memory:           " << props.totalGlobalMem / mb << "mb\n";
        std::cout << "  Shared memory:           " << props.sharedMemPerBlock / kb << "kb\n";
        std::cout << "  Constant memory:         " << props.totalConstMem / kb << "kb\n";
        std::cout << "  Block registers:         " << props.regsPerBlock << "\n";
        std::cout << "  ECC enabled:             " << props.ECCEnabled << "\n";
        std::cout << "  memoryBusWidth           " << props.memoryBusWidth << "\n";
        std::cout << "  l2CacheSize              " << props.l2CacheSize << "\n";
        std::cout << "  persistingL2CacheMaxSize:" << props.persistingL2CacheMaxSize << "\n";
        std::cout << "\n";

        std::cout << "  Warp size:         " << warpsize << "\n";
        std::cout << "  Threads per block: " << props.maxThreadsPerBlock << "\n";
        std::cout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1] << ", " << props.maxThreadsDim[2] << " ]\n";
        std::cout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " << props.maxGridSize[1] << ", " << props.maxGridSize[2] << " ]\n";
        std::cout << std::endl;
    }
}



//multiplier to change bitmask 0b____QRST to 0b___Q'___R'___S'___T
static constexpr auto VarMaskSplitter = 0x00'20'40'81;


__device__ [[nodiscard]] uint32_t GetFollower(const int BallotMask) {
    return 31 - __clz(BallotMask);
}

struct xorshiftM_t {
    uint32_t state;

    __host__ __device__ xorshiftM_t() {
        state = 32657 * (ThreadId() + 1);
    }

    __host__ __device__ xorshiftM_t(uint32_t init) : state(init) {
        if (state == 0) { state = 32657 * (ThreadId() + 1); }
    }

    __host__ __device__ uint32_t next() {
        assert(0 != state);
        state ^= (state << 13);
        state ^= (state >> 17);
        state ^= (state << 5);
        const auto result = state * 1597334677;
        state ^= 1; //escape zero state;
        return result;
    }

    __host__ __device__ uint32_t NextInt(uint32_t max) {
        auto result = next();
        return (result % max);
    }

    template <uint32_t max>
    __host__ __device__ uint32_t NextInt() {
        auto result = next();
        return (result % max);
    }

    __host__ __device__ float NextFloat() {
        return 2.3283064365387e-10f * next();
    }

    __host__ __device__ float NextFloat(float max) {
        return NextFloat() * max;
    }

    template <int LogLevel>
    __host__ __device__ void show() const {
        if constexpr (LogLevel > 2) {
            printf("xorshift.state = %i($%x)", state, state);
        }
    }
};

struct xorshiftPlus_t {
    uint32_t state;
    __host__ __device__ uint32_t next() {
        auto x = state;
        state ^= (state << 13);
        state ^= (state >> 17);
        state ^= (state << 5);
        return state + (x & 0xFFFF);
    }

    template <uint32_t max>
    __host__ __device__ uint32_t next() {
        return next() % max;
    }

    template <uint32_t max>
    __host__ __device__ uint32_t NextInt() {
        return next() % max;
    }

    __host__ __device__ uint32_t NextInt(uint32_t max) {
        return next() % max;
    }

    __host__ __device__ float NextFloat() {
        return 2.3283064365387e-10f * next();
    }

    __host__ __device__ float NextFloat(int Multiplier) {
        auto result = 2.3283064365387e-10f * next();
        result *= Multiplier;
        return result;
    }

    __host__ __device__ xorshiftPlus_t(uint32_t init) : state(init) {
        if (0 == state) { state = 0xcafebabe; }
    }
    __host__ __device__ xorshiftPlus_t() {}
};

static const int LogAlways = 0;
static const int LogNormal = 1;
static const int LogVerbose = 2;
static const int LogVeryVerbose = 3;

#define Log(AskedLogLevel, format, ...) \
static_assert((uint32_t)(LogLevel) <= 3, "LogLevel out of bounds"); \
static_assert((uint32_t)(AskedLogLevel) <= 3, "requested Loglevel out of bounds"); \
do {                                              \
    if constexpr((LogLevel) >= (AskedLogLevel)) { \
        printf("c " format "\n", __VA_ARGS__); \
    }                                             \
} while (0)

#define LogIf(AskedLogLevel, Predicate, format, ...) \
static_assert((uint32_t)(LogLevel) <= 3, "LogLevel out of bounds"); \
static_assert((uint32_t)(AskedLogLevel) <= 3, "requested Loglevel out of bounds"); \
do {                                              \
    if constexpr((LogLevel) >= (AskedLogLevel)) { \
        if (Predicate) { printf("c " format "\n", __VA_ARGS__); } \
    }                                             \
} while (0)

//Align and insert a bit field from source into dest. bit gives the starting bit position for the insertion, and num_bits gives the bit field length in bits. 
__host__ __device__ [[nodiscard]] unsigned bfi(unsigned source, unsigned dest, unsigned bit, unsigned num_bits) {
    unsigned result;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
    asm("bfi.b32 %0, %1, %2, %3, %4;" :
    "=r"(result) : "r"(source), "r"(dest), "r"(bit), "r"(num_bits));
#else
    if (bit + num_bits > 32) { num_bits = 32 - bit; }
    unsigned mask = ((1 << num_bits) - 1) << bit;
    result = dest & (~mask);
    result |= mask & (source << bit);
#endif
    return result;
}

//https://github.com/moderngpu/moderngpu/blob/master/src/moderngpu/intrinsics.hxx
template <int num_bits = 1>
__host__ __device__ [[nodiscard]] unsigned bfe(const unsigned x, const unsigned bit) {
    unsigned result;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
    asm("bfe.u32 %0, %1, %2, %3;" :
    "=r"(result) : "r"(x), "r"(bit), "r"(num_bits));
#else
    result = ((1 << num_bits) - 1) & (x >> bit);
#endif
    return result;
}

#define Bin "%c%c%c%c%c%c%c%c"
#define ToBin(byte)  \
  (byte & 0x80 ? '1' : '0'), \
  (byte & 0x40 ? '1' : '0'), \
  (byte & 0x20 ? '1' : '0'), \
  (byte & 0x10 ? '1' : '0'), \
  (byte & 0x08 ? '1' : '0'), \
  (byte & 0x04 ? '1' : '0'), \
  (byte & 0x02 ? '1' : '0'), \
  (byte & 0x01 ? '1' : '0') 


#endif