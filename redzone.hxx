#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "Macros.hxx"  
#include <thrust/remove.h>
#include <thrust/execution_policy.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include "DataStruct.hxx"
namespace cg = cooperative_groups;

template <typename T>
__device__ int ThrustCompact(T* data, const int Length) {
    __syncthreads();
    const auto end = thrust::remove(thrust::device, data, data + Length, -1);
    __syncthreads();
    return end - data;
}

template <int MaxRemoveCount, typename T>
struct Orphan_t {
    int DestOrphans[(MaxRemoveCount + 1) / 2];
    T SourceOrphans[(MaxRemoveCount + 1) / 2];
};

template <int BlockCount>
struct BlockStorage_t {
    BlockStorage_t<BlockCount>* self;
    uint64_t Mask1;
    uint64_t Mask2;
    uint32_t SourceOrphanStart;
    uint32_t DestOrphanStart;
    //uint32_t ReadPos;
    uint32_t DestCounts[BlockCount * 2];
    uint32_t SourceCounts[BlockCount * 2];
    uint64_t PhaseTimes[3];
    #ifdef _DEBUG
    uint32_t Totals[2];
    #endif

    #ifdef __CUDA_ARCH__
    __device__ BlockStorage_t() {}
    #else
    __host__ BlockStorage_t() {

        CUDA(cudaMalloc(&self, sizeof(BlockStorage_t<BlockCount>)));
        Mask1 = 0;
        Mask2 = 0;
        SourceOrphanStart = 0;
        DestOrphanStart = 0;
        //ReadPos = 0;
        CUDA(cudaMemset(&self->Mask1, 0, sizeof(BlockStorage_t<BlockCount>) - sizeof(int*)));
        CUDA(cudaMemcpy(&self->self, &self, sizeof(BlockStorage_t<BlockCount>*), cudaMemcpyHostToDevice));
    }
    #endif
    __device__ [[nodiscard]] constexpr auto SollMask() { return (uint64_t(1) << BlockCount) - 1; }
    
   
    __device__ void Wait() {
        constexpr auto maxwaittime = 0xFFFF;
        const auto Bit = mod<64>(BlockId());
        auto waittime = 1;
        //__syncthreads();
        if (ThreadId() == 0) {
            do {
                __nanosleep(waittime);
                waittime = std::max((waittime * 2) + 1, maxwaittime); //even on overflow, it is still a valid time
            } while (atomicOr((unsigned long long int*)&Mask1, (unsigned long long int)(1ull << Bit)) != SollMask());

            //reset the mask back to zero.
            waittime = 1;
            do {} while (atomicOr((unsigned long long int*)&Mask2, (unsigned long long int)(1ull << Bit)) != SollMask());

            //reset mask1 back to zero.
            do {} while (atomicAnd((unsigned long long int*)&Mask1, (unsigned long long int)(~(1ull << Bit))) != 0);
            //reset mask2 back to zero.
            do {} while (atomicAnd((unsigned long long int*)&Mask2, (unsigned long long int)(~(1ull << Bit))) != 0);
        }
        __syncthreads();
    }
};



//this code assumes there are no duplicate entries in Removals
template <int MaxRemoveCount, int WarpCount, bool DebugPrint, typename T>
__device__ void RemoveFromList(T* List, const int Length, int* Removals, const int RemoveCount, Orphan_t<MaxRemoveCount, T>& Orphans) {
    constexpr auto InvalidSource = 0x8000'0000;
    //somewhat confusingly we store the validity of the sources in the destination, because that is more efficient
    const auto GetIsSourceValid = [InvalidSource](const int dest)-> bool { return (dest & InvalidSource) == 0; };
    const auto DestIndex = [InvalidSource](const int dest)-> int { return  dest & (~InvalidSource); };
    const auto CleanDest = [InvalidSource](const int dest)-> int { return  dest ^ InvalidSource; };

    if (RemoveCount == Length || RemoveCount == 0) { return; }

    //Current GPUs cannot have more than 32 warps per block
    constexpr auto MaxBlockSize = GPUBlockSize;
    __shared__ int PrefixStartsSource[(MaxBlockSize / warpsize) + 1]; //exclusive scan of the number of orphans per warp
    __shared__ int PrefixStartsDest[(MaxBlockSize / warpsize) + 1]; //exclusive scan of the number of orphans per warp
    //Calculate offsets for the warps to store their orphans and corrections
    const auto PrefixOffsets = [](int* PrefixStarts, int Offset, int StartCount) -> int {
        assert1(__ballot_sync(Everyone, 1) == Everyone);
        assert1(__syncthreads_count(1) == BlockSize());
        PrefixStarts[0] = StartCount;
        if constexpr (WarpCount == 1) {
            PrefixStarts[WarpCount] = Offset + StartCount;
        }
        if constexpr (WarpCount > 1) {
            PrefixStarts[WarpId() + 1] = Offset;
            __syncthreads();
            if (WarpId() == 0) {
                auto OrphanSum = PrefixStarts[LaneId() + 1];
                #pragma unroll
                for (auto i = 1; i < WarpCount; i *= 2) {
                    const auto Add = __shfl_up_sync(Everyone, OrphanSum, i);
                    if (LaneId() >= i) { OrphanSum += Add; }
                }
                PrefixStarts[LaneId() + 1] = (OrphanSum + StartCount);
            }
        }
        __syncthreads();
        return PrefixStarts[WarpCount]; //start point for the next iteration
    };


    const auto RedZoneStart = Length - RemoveCount;
    __syncthreads();
    //Phase 1. The redzone contains the source items that will be used to fill the holes
    //The removals contain the destination items that will be removed
    //If a redzone item is itself to be removed, then it is not a valid source item
    //Mark all those items as invalid.
    printonceif(DebugPrint, "Start Phase 1: mark all removal items inside the redzone\n");
    printonceif(DebugPrint, "RedZoneStart = %i, FalseCount = %i, removeCount = %i\n", RedZoneStart, Length, RemoveCount);
    for (auto i = ThreadId(); i < RemoveCount; i += BlockSize()) {
        const auto dest = DestIndex(Removals[i]);
        if (dest >= RedZoneStart) {
            printif(DebugPrint, "Item[%i]:$%x(%i) Source[%i] = removed\n", i, dest, dest, dest - RedZoneStart);
            assert2((Removals[dest - RedZoneStart] & InvalidSource) == 0, "Oops, trying to remove an item[%i] that is already removed", dest - RedZoneStart);
            Removals[dest - RedZoneStart] |= InvalidSource;  //mark the invalid status of the source in the destination for performance reasons
        }
    } // for
    __syncthreads();

    printonceif(DebugPrint, "Start Phase 2, Move valid sources to valid dest, save invalid sources/dest. in the orphan list");
    //Phase 2. We have sources in the redzone (Z), and destination (indices) in r
    //both the source and the destination can be either valid or invalid
    //This means we have 4 cases to deal with
    // 1. invalid source Z, invalid destination r -> do nothing
    // 2. valid source, invalid destination       -> store the source in the orphan list
    // 3. invalid source, valid destination       -> store the destination in the correction list
    // 4. valid source, valid destination         -> move the source to the destination  
    auto SourceOrphanCount = 0;
    auto DestOrphanCount = 0;
    const auto MaxRuns = (RemoveCount + BlockSize() - 1) / BlockSize();
    for (auto run = 0, i = ThreadId(); run < MaxRuns; run++, i += BlockSize()) {  //we need all threads active to create the prefix sum
        auto dest = uint32_t(RedZoneStart | InvalidSource); //mark out of range items as source + dest = invalid to prevent them from being processed
        T source;
        if (i < RemoveCount) {
            dest = Removals[i];                    //destination
            source = List[RedZoneStart + i];       //the item to fill the hole with
            printif(DebugPrint, "************ in the redzone ************ i= %i, removeIndex(dest) = %i, isDeleted(dest) = %i\n", i, DestIndex(dest), !GetIsSourceValid(dest));
            printif(DebugPrint, "Source: List[RedZoneStart(%i) + %i=%i] = %i\n", RedZoneStart, i, RedZoneStart + i, source);
        }
        #ifdef _DEBUG
        auto _case = 4; //assume invalid source and invalid destination
        #endif
        const auto isDestValid = (DestIndex(dest) < RedZoneStart);
        const auto isSourceValid = GetIsSourceValid(dest);
        const auto isSourceOrphan = (isSourceValid && (!isDestValid));   //source is valid, but destination is invalid -> store the source as orphan
        const auto SourceOrphanMask = __ballot_sync(Everyone, isSourceOrphan);
        SourceOrphanCount = PrefixOffsets(PrefixStartsSource, __popc(SourceOrphanMask), SourceOrphanCount);
        const auto SourceOrphanIndex = LanesBefore<lt>(SourceOrphanMask) + PrefixStartsSource[WarpId()];
        if (isSourceOrphan) {
            #ifdef _DEBUG
            _case = 1;
            #endif
            assert(!isDestValid);
            assert(isSourceValid);
            Orphans.SourceOrphans[SourceOrphanIndex] = source;
            printif(DebugPrint, "SourceOrphans[%i] = Source(%i)\n", SourceOrphanIndex, source);
        }

        //If the destination is valid, but the source is not, then store the dest as orphan
        const auto isDestOrphan = ((!isSourceValid) && isDestValid);
        const auto DestOrphanMask = __ballot_sync(Everyone, isDestOrphan);
        DestOrphanCount = PrefixOffsets(PrefixStartsDest, __popc(DestOrphanMask), DestOrphanCount);
        const auto DestOrphanIndex = LanesBefore<lt>(DestOrphanMask);

        if (isDestOrphan) {
            //We need to store these items, because some of them might reference items that will get orphaned in future iterations
            assert1((dest & InvalidSource) == InvalidSource);
            assert1(!isSourceOrphan);
            assert1(!isSourceValid);
            #ifdef _DEBUG
            _case = 2;
            #endif
            Orphans.DestOrphans[PrefixStartsDest[WarpId()] + DestOrphanIndex] = CleanDest(dest);
            printif(DebugPrint, "DestOrphans[%i] = %i\n", PrefixStartsDest[WarpId()] + DestOrphanIndex, CleanDest(dest));
        }

        const auto isAllowed = (isSourceValid && isDestValid);
        if (isAllowed) {
            #ifdef _DEBUG
            _case = 3;
            #endif
            List[dest] = source; //no need to strip Removed from source.
            printif(DebugPrint, "Dest[%i] <= %i\n", dest, source);

        }
        #ifdef _DEBUG
        if (!isSourceValid && !isDestValid) { assert1(4 == _case); }
        #endif
                //printif((!isSourceValid && !isDestValid), "Case 4: dest = %i, source = %i\n", DestIndex(dest), source);
    } //for phase 2
    __syncthreads();

    //All the normal items have been processed, now we need to process the orphaned items
    //Phase 3, pair up all the corrections with the orphans
    printonceif(DebugPrint, "************ Entering phase 3: pair up SourceOrphans with DestOrphans ************\n");
    assert2(isEqual(DestOrphanCount, SourceOrphanCount), "");
    //printif(DestOrphanCount != SourceOrphanCount, "DestOrphanCount = %i, SourceOrphanCount = %i\n", DestOrphanCount, SourceOrphanCount);
    for (auto i = ThreadId(); i < DestOrphanCount; i += BlockSize()) {
        const auto dest = Orphans.DestOrphans[i];
        assert1((dest & InvalidSource) == 0);
        const auto source = Orphans.SourceOrphans[i];
        List[dest] = source;
        printif(DebugPrint, "Dest[%i] = %i = SourceOrphans[%i]\n", dest, source, i);
    }
    printonceif(DebugPrint, "New length = %i\n", Length - RemoveCount);
    __syncthreads();
}

//################################################################################################
//################################################################################################

//this code assumes there are no duplicate entries in Removals
template <int WarpCount, bool DebugPrint, typename T>
__device__ void RemoveFromList2(T* List, const int Length, int* Removals, const int RemoveCount, float* TestTime) {
    uint64_t StartTime;
    uint64_t PrevTime;
    if (ThreadId() == 0) {
        PrevTime = clock64();
        StartTime = PrevTime;
    }
    constexpr auto blocksize = WarpCount * warpsize;
    const auto RedZoneStart = Length - RemoveCount;
    constexpr auto prefetchcount = 4;
    assert(blocksize == BlockSize());
    auto pipeline = cuda::make_pipeline();
    //prime the pump, read the first part of list into shared memory
    __shared__ int Buffer[blocksize * 2 * prefetchcount];
    T* ListBuffer = &Buffer[0];
    int* RemoveBuffer = &Buffer[blocksize * prefetchcount];

    constexpr auto InvalidSource = 0x8000'0000;
    if (RemoveCount == Length || RemoveCount == 0) { return; }

    //Current GPUs cannot have more than 32 warps per block
    constexpr auto MaxBlockSize = GPUBlockSize;
    __shared__ int PrefixStartsSource[(MaxBlockSize / warpsize) + 1]; //exclusive scan of the number of orphans per warp
    __shared__ int PrefixStartsDest[(MaxBlockSize / warpsize) + 1]; //exclusive scan of the number of orphans per warp
    //Calculate offsets for the warps to store their orphans and corrections
    const auto PrefixOffsets = [](int* PrefixStarts, int Offset, int StartCount) -> int {
        assert1(__ballot_sync(Everyone, 1) == Everyone);
        assert1(__syncthreads_count(1) == blocksize);
        PrefixStarts[0] = StartCount;
        if constexpr (WarpCount == 1) {
            PrefixStarts[WarpCount] = Offset + StartCount;
        }
        if constexpr (WarpCount > 1) {
            PrefixStarts[WarpId() + 1] = Offset;
            __syncthreads();
            if (ThreadId() < warpsize) {
                auto OrphanSum = PrefixStarts[ThreadId() + 1];
                #pragma unroll
                for (auto i = 1; i < WarpCount; i *= 2) {
                    const auto Add = __shfl_up_sync(Everyone, OrphanSum, i);
                    if (ThreadId() >= i) { OrphanSum += Add; }
                }
                PrefixStarts[ThreadId() + 1] = (OrphanSum + StartCount);
            }
        }
        __syncthreads();
        return PrefixStarts[WarpCount]; //start point for the next iteration
    };

    __syncthreads();
    //111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
    //Phase 1. The redzone contains the source items that will be used to fill the holes
    //The removals contain the destination items that will be removed
    //If a redzone item is itself to be removed, then it is not a valid source item
    //Mark all those items as invalid.
    printonceif(DebugPrint, "Start Phase 1: mark all removal items inside the redzone\n");
    printonceif(DebugPrint, "RedZoneStart = %i, FalseCount = %i, removeCount = %i\n", RedZoneStart, Length, RemoveCount);
    for (auto i = ThreadId(); i < RemoveCount; i += blocksize) {
        const auto dest = Removals[i];
        if (dest >= RedZoneStart) {
            printif(DebugPrint, "Item[%i]:$%x(%i) Source[%i] = removed\n", i, dest, dest, dest - RedZoneStart);
            assert2((Removals[dest - RedZoneStart] & InvalidSource) == 0, "Oops, trying to remove an item[%i] that is already removed", dest - RedZoneStart);
            //atomicOr(&Removals[dest - RedZoneStart], InvalidSource);  //mark the invalid status of the source in the destination for performance reasons
            List[dest] = InvalidSource;
        }
    } // for
    __syncthreads();
    if (ThreadId() == 0) {
        const auto Time1 = clock64();
        TestTime[1] = float(Time1 - PrevTime) / 1000.0f;
        PrevTime = Time1;
    }

    const auto Prefetch1 = [&](const int run) {
        pipeline.producer_acquire();
        const auto Offset = run * blocksize;
        const auto start = Offset + (ThreadId());
        const int size = std::max(std::min((RemoveCount - start), 1), 0);
        if (size) {
            const auto r = mod<prefetchcount>(run);
            memcpy_async(&ListBuffer[ThreadId() + (r * blocksize)], &List[RedZoneStart + start], sizeof(T), pipeline);
            memcpy_async(&RemoveBuffer[ThreadId() + (r * blocksize)], &Removals[start], sizeof(int), pipeline);
        }
        pipeline.producer_commit();
    };

    //DoPrefetch(0);
    #pragma unroll
    for (auto i = 0; i < prefetchcount; i++) { Prefetch1(i); }


    printonceif(DebugPrint, "Start Phase 2, Move valid sources to valid dest, save invalid sources/dest. in the orphan list");
    //2222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222
    //Phase 2. We have sources in the redzone (Z), and destination (indices) in r
    //both the source and the destination can be either valid or invalid
    //This means we have 4 cases to deal with
    // 1. invalid source Z, invalid destination r -> do nothing
    // 2. valid source, invalid destination       -> store the source in the orphan list
    // 3. invalid source, valid destination       -> store the destination in the correction list
    // 4. valid source, valid destination         -> move the source to the destination  
    auto SourceOrphanCount = 0;
    auto DestOrphanCount = 0;
    const auto MaxRuns = (RemoveCount + blocksize - 1) / blocksize;
    for (auto run = 0, i = ThreadId(); run < MaxRuns; run++, i += blocksize) {  //we need all threads active to create the prefix sum
        auto dest = uint32_t(RedZoneStart); //mark out of range items as source + dest = invalid to prevent them from being processed
        T source = InvalidSource;

        cuda::pipeline_consumer_wait_prior<prefetchcount - 1>(pipeline);

        if (i < RemoveCount) {
            dest = RemoveBuffer[mod< blocksize * prefetchcount>(i)];
            source = ListBuffer[mod< blocksize * prefetchcount>(i)];
            printif(DebugPrint, "************ in the redzone ************ i= %i, removeIndex(dest) = %i, isDeleted(source) = %i\n", i, dest, InvalidSource == source);
            printif(DebugPrint, "Source: List[RedZoneStart(%i) + %i=%i] = %i\n", RedZoneStart, i, RedZoneStart + i, source);
        }

        Prefetch1(run + prefetchcount);


        #ifdef _DEBUG
        auto _case = 4; //assume invalid source and invalid destination
        #endif
        const auto isDestValid = (dest < RedZoneStart);
        const auto isSourceValid = source != InvalidSource;
        const auto isSourceOrphan = (isSourceValid && (!isDestValid));   //source is valid, but destination is invalid -> store the source as orphan
        const auto SourceOrphanMask = __ballot_sync(Everyone, isSourceOrphan);
        SourceOrphanCount = PrefixOffsets(PrefixStartsSource, __popc(SourceOrphanMask), SourceOrphanCount);
        const auto SourceOrphanIndex = LanesBefore<lt>(SourceOrphanMask) + PrefixStartsSource[WarpId()];


        if (isSourceOrphan) {
            #ifdef _DEBUG
            _case = 1;
            #endif
            assert(!isDestValid);
            assert(isSourceValid);
            List[RedZoneStart + SourceOrphanIndex] = source;
            printif(DebugPrint, "SourceOrphans[%i] = Source(%i)\n", SourceOrphanIndex, source);
        }

        //If the destination is valid, but the source is not, then store the dest as orphan
        const auto isDestOrphan = ((!isSourceValid) && isDestValid);
        const auto DestOrphanMask = __ballot_sync(Everyone, isDestOrphan);
        DestOrphanCount = PrefixOffsets(PrefixStartsDest, __popc(DestOrphanMask), DestOrphanCount);
        const auto DestOrphanIndex = LanesBefore<lt>(DestOrphanMask) + PrefixStartsDest[WarpId()];

        if (isDestOrphan) {
            //We need to store these items, because some of them might reference items that will get orphaned in future iterations
            assert1(InvalidSource == source);
            assert1(!isSourceOrphan);
            assert1(!isSourceValid);
            //assert1(DestOrphanIndex >= DestIndexOffset);
            #ifdef _DEBUG
            _case = 2;
            #endif
                        //DestOrphans[DestOrphanIndex - DestIndexOffset] = CleanDest(dest);
            Removals[DestOrphanIndex] = dest;
            printif(DebugPrint, "DestOrphans[%i] = %i\n", PrefixStartsDest[WarpId()] + DestOrphanIndex, dest);
        }

        const auto isAllowed = (isSourceValid && isDestValid);
        if (isAllowed) {
            #ifdef _DEBUG
            _case = 3;
            #endif
            List[dest] = source; //no need to strip Removed from source.
            printif(DebugPrint, "Dest[%i] <= %i\n", dest, source);

        }
        #ifdef _DEBUG
        if (!isSourceValid && !isDestValid) { assert1(4 == _case); }
        #endif
                //printif((!isSourceValid && !isDestValid), "Case 4: dest = %i, source = %i\n", DestIndex(dest), source);
    } //for phase 2
    __syncthreads();
    if (ThreadId() == 0) {
        const auto Time2 = clock64();
        TestTime[2] = float(Time2 - PrevTime) / 1000.0f;
        PrevTime = Time2;
    }
    //33333333333333333333333333333333333333333333333333333333333333333333333333333333
    //All the normal items have been processed, now we need to process the orphaned items
    //Phase 3, pair up all the corrections with the orphans
    printonceif(DebugPrint, "************ Entering phase 3: pair up SourceOrphans with DestOrphans ************\n");
    assert2(isEqual(DestOrphanCount, SourceOrphanCount), "DestOrphanCount = %i, SourceOrphanCount = %i, D-S = %i", DestOrphanCount, SourceOrphanCount, (DestOrphanCount - SourceOrphanCount));

    for (auto i = ThreadId(); i < DestOrphanCount; i += blocksize) {
        //get the data into the orphan list asynchronously
        //use memcpy_async
        //const auto dest = (i < DestIndexOffset)? Removals[i] : DestOrphans[i - DestIndexOffset];
        const auto dest = Removals[i];
        assert1((dest & InvalidSource) == 0);
        //const auto source = (i < SourceIndexOffset)? List[RedZoneStart + i] : SourceOrphans[i - SourceIndexOffset];
        const auto source = List[RedZoneStart + i];
        List[dest] = source;
        assert1(dest < RedZoneStart);
        printif(DebugPrint, "P3: Dest[%i] = %i = SourceOrphans[%i]\n", dest, source, i);
    }
    printonceif(DebugPrint, "New length = %i\n", Length - RemoveCount);
    __syncthreads();
    if (ThreadId() == 0) {
        const auto Time3 = clock64();
        TestTime[3] = float(Time3 - PrevTime) / 1000.0f;
        TestTime[0] = float(Time3 - StartTime) / 1000.0f;
        //printf("OrphanCount = %i, T1: %.2f, T2: %.2f, T3: %.2f, TT: %.2f\n", DestOrphanCount, TestTime[1], TestTime[2], TestTime[3], TestTime[0]);
    }
}

template <int WarpCount, bool DebugPrint, typename T>
__global__ void RedzoneOneBlock(T* List, const int Length, int* Removals, const int RemoveCount, float* TestTime) {
    RemoveFromList2<WarpCount, DebugPrint>(List, Length, Removals, RemoveCount, TestTime);
}


//################################################################################################
//################################################################################################


template <int WarpCount, int BlockCount, int prefetchcount>
__global__ void TestGPUWide(int* A, int* sizeA, int* R, int* sizeR, BlockStorage_t<BlockCount>* BlockStore, float* TestTime) {
    uint64_t Time = clock64();

    #ifdef _DEBUG
    constexpr auto DebugPrint = false;
    #else
    constexpr auto DebugPrint = false;
    #endif
    RemoveFromList_GPUWide<WarpCount, BlockCount, prefetchcount, DebugPrint>(A, *sizeA, R, *sizeR, *BlockStore);
    BlockStore->Wait();
    Time = clock64() - Time;
    float ftime = float(Time) / 1000.0f;
    TestTime[0] = ftime;
    for (auto i = 0; i < 3; i++) { TestTime[i + 1] = float(BlockStore->PhaseTimes[i]) / 1000.0f; }
    //printf("n = 1 << %i, k = %i%%, r = %i%%, time1 = %fus, time2 = %fus, time3 = %f\n", __ffs(*sizeA), 2, TestTime[1], TestTime[2], TestTime[3]);
    //printonce("RemoveFromList: Blocksize = %i, Oldsize: %i, remove: %i, New size = %i took %.2f us\n", blockDim.x, *sizeA, *sizeR, *sizeA - *sizeR, ftime);
}

//Return the index of the item to work on
//given a list of counts
template <int warpcount, int blockcount>
__device__ int LoadBalance(int run, const int* SharedCount) {
    constexpr auto blocksize = warpcount * warpsize;
    constexpr auto gridsize = blockcount * blocksize;
    constexpr auto n = blockcount; //we cannot have more chunks than we have blocks in the grid
    const auto Start = (run * gridsize) + (BlockId() * blocksize) + (WarpId() * warpsize); //example: b:1 w:1 -> start = 64+32 = 96
    if (Start > SharedCount[n]) { return -1; }
    assert1(0 == SharedCount[0]);
    SharedCount++; //skip the first item, because it is always zero
    constexpr auto runs = DivUp<warpsize>(blockcount);
    __shared__ int Storage[warpcount][runs];
    const auto w = WarpId();
    const auto tid = LaneId() + Start;

    //Find the smallest index that is >= than start
    auto MinIndex = MAXINT32;
    #pragma unroll
    for (auto r = 0, lid = int(LaneId()); r < runs; r++, lid += warpsize) {
        if ((lid < n) && (SharedCount[lid] >= Start)) { MinIndex = lid; }
        MinIndex = __reduce_min_sync(Everyone, MinIndex);
        if constexpr (runs > 1) { Storage[w][r] = MinIndex; }
    }
    if constexpr (runs > 1) {
        __syncwarp();
        if (LaneId() < runs) { MinIndex = Storage[w][LaneId()]; }
        MinIndex = __reduce_min_sync(Everyone, MinIndex);
    }
    assert1(MinIndex >= 0);
    assert1(MinIndex < n);
    constexpr auto LastIndex = (n - 1);
    const auto FirstIndex = MinIndex;

    //const auto Range = LastIndex - FirstIndex;
    auto Index = FirstIndex;
    //auto Offset = 1;// 1u << (31 - __clz(Range));
    if (tid >= SharedCount[LastIndex]) { Index = -1; } else {
        //linear search for the index
        bool GetNext;
        do {   //we also need to increase index if needed
            GetNext = (tid >= SharedCount[Index]);
            //const auto GetPrev = (tid < SharedCount[Index - 1]);
            //printif((GetNext && GetPrev), "tid = %i, SharedCount[%i] = %i, SharedCount[%i-1] = %i", tid, Index, SharedCount[Index], Index, SharedCount[Index - 1]);
            Index += GetNext; //goto previous item if not within bounds
            //Offset >>= 1;
        } while (GetNext); //Keep going until no more threads have incorrect entries
    }
    return Index;
}

template <int WarpCount, bool do_prefetch, typename T>
__device__ void StreamCompact_OneBlock(T* List, const int Length) {
    #ifdef _DEBUG
    __shared__ int TestIndex;
    TestIndex = 0;
    #endif
        //printif(ThreadId() < 100, "BlockDim = %i, WarpCount = %i, List[%i] = %i\n", blockDim.x, WarpCount, DeviceThreadId(), List[DeviceThreadId()]);
    constexpr auto blocksize = WarpCount * warpsize;
    constexpr auto prefetchcount = 4;
    __shared__ T ListBuffer[blocksize * prefetchcount];
    assert(blocksize == BlockSize());
    auto pipeline = cuda::make_pipeline();
    //prime the pump, read the first part of list into shared memory
    const auto Prefetch = [&](int run) {
        pipeline.producer_acquire();
        const auto start = (run * blocksize) + ThreadId();
        const auto size = std::max(std::min(Length - start, 1), 0);
        if (size) {
            const auto r = mod<prefetchcount>(run);
            memcpy_async(&ListBuffer[ThreadId() + (r * blocksize)], &List[start], sizeof(T), pipeline);
        }
        pipeline.producer_commit();
    };

    __shared__ int OffsetInput[warpsize + 1];
    __shared__ int OffsetOutput[warpsize + 1];
    OffsetInput[0] = 0;
    OffsetOutput[0] = 0;
    OffsetOutput[LaneId() + 1] = 0;

    const auto prefix = [&](const int WarpOffset) {
        __syncthreads();
        if (LaneId() == 0) {
            OffsetInput[WarpId()] = WarpOffset;
        }
        __syncthreads();
        if (WarpId() == 0) {
            auto OrphanSum = OffsetInput[LaneId()];
            const auto StartCount = OffsetOutput[WarpCount];
            #pragma unroll
            for (auto i = 1; i < WarpCount; i *= 2) {
                const auto Add = __shfl_up_sync(Everyone, OrphanSum, i);
                if (LaneId() >= i) { OrphanSum += Add; }
            }
            OffsetOutput[LaneId() + 1] = OrphanSum + StartCount;
            OffsetOutput[0] = StartCount;
        }
        __syncthreads();
        return OffsetOutput[WarpId()];
    };

    if constexpr (do_prefetch) {
        for (auto i = 0; i < prefetchcount; i++) { Prefetch(i); }
    }

    __shared__ T Buffer[blocksize];
    const auto MaxRuns = DivUp<blocksize>(Length);
    for (auto i = ThreadId(), run = 0; run < MaxRuns; i += blocksize, run++) {
        T data;
        if constexpr (do_prefetch) {
            const auto r = mod<prefetchcount>(run);
            cuda::pipeline_consumer_wait_prior<prefetchcount - 1>(pipeline);
            data = ListBuffer[ThreadId() + (r * blocksize)];
            assert(data == List[i]);
            Prefetch(run + prefetchcount);
        } else {
            data = List[i];
        }
        const auto keep = (data != T(-1));
        const auto KeepMask = __ballot_sync(Everyone, keep);
        const auto count = __popc(KeepMask);
        const auto LaneOffset = LanesBefore(KeepMask);
        const auto WarpOffset = prefix(count);
        if (keep) {
            List[WarpOffset + LaneOffset] = data;
        }
    }
}

template <int WarpCount, bool do_prefetch, bool init_stencil, typename T>
__global__ void TestO_N_Removal_OneBlock(T* List, const int Length, int* Removals, const int Rlength, float* Time) {
    //printif(DeviceThreadId() == 0, "TestO_N_Removal_OneBlock: Length = %i\n", Length);
    const auto StartTime = clock64();
    if constexpr (init_stencil) {
        for (auto i = DeviceThreadId(); i < Rlength; i += BlockSize()) {
            const auto index = Removals[i];
            List[index] = T(-1);
        }
        __syncthreads();
    }
    const auto Time2 = clock64();
    StreamCompact_OneBlock<WarpCount, do_prefetch>(List, Length);
    __syncthreads();
    const auto EndTime = clock64();
    Time[0] = float(EndTime - StartTime) / 1000.0f;
    Time[1] = float(Time2 - StartTime) / 1000.0f;
    Time[2] = float(EndTime - Time2) / 1000.0f;
}

//this code assumes there are no duplicate entries in Removals
template <int WarpCount, int BlockCount, int prefetchcount, bool DebugPrint, typename T>
__device__ void RemoveFromList_GPUWide(T* List, const int Length, int* Removals, const int RemoveCount, BlockStorage_t<BlockCount>& BlockStore) {
    auto g = cg::this_grid();
    if (DeviceThreadId() == 0) { BlockStore.PhaseTimes[0] = clock64(); }

    constexpr auto blocksize = WarpCount * warpsize;
    constexpr auto gridsize = blocksize * BlockCount; //thread count in all blocks
    assert(blocksize == BlockSize());
    assert(g.is_valid());
    assert(gridsize == g.size());
    static_assert(sizeof(T) == sizeof(int));
    const auto RedZoneStart = Length - RemoveCount;
    //constexpr auto prefetchcount = 5;

    auto pipeline = cuda::make_pipeline();
    extern __shared__ int InputBuffer[];
    T* ListBuffer = &InputBuffer[0];                               // 1024*4*4 = 16KB
    int* RemoveBuffer = &InputBuffer[blocksize * prefetchcount];   // 1024*4*4 = 16KB

    constexpr auto InvalidSource = 0x8000'0000;
    //somewhat confusingly we store the validity of the sources from List in the destination from Removals, because that is more efficient
    const auto GetIsSourceValid = [InvalidSource](const int dest)-> bool { return (dest & InvalidSource) == 0; };
    const auto DestIndex = [InvalidSource](const int dest)-> int { return  dest & (~InvalidSource); };
    const auto CleanDest = [InvalidSource](const int dest)-> int { return  dest ^ InvalidSource; };

    if (RemoveCount == Length || RemoveCount == 0) { return; }

    __shared__ int PrefixStartsSource[warpsize + 1]; //exclusive scan of the number of orphans per warp
    __shared__ int PrefixStartsDest[warpsize + 1]; //exclusive scan of the number of orphans per warp

    //Calculate offsets for the warps to store their orphans and corrections
    const auto PrefixOffsets = [](int* PrefixStarts, int Offset, int StartCount) -> int {
        assert1(__ballot_sync(Everyone, 1) == Everyone);
        assert1(__syncthreads_count(1) == blocksize);
        PrefixStarts[0] = StartCount;
        static_assert(WarpCount > 0);
        #ifdef _DEBUG
        if (__match_any_sync(Everyone, Offset) != Everyone) {
            assert2(__match_any_sync(Everyone, Offset) == Everyone, "Expected Offset all equal, Offset = %i OffsetMask = %x", Offset, __ballot_sync(Everyone, __shfl_sync(Everyone, Offset, 0) == Offset));
        }
        #endif
        if constexpr (WarpCount == 1) {
            const auto result = Offset + StartCount;
            PrefixStarts[WarpCount] = result;
            return result;
        }
        PrefixStarts[WarpId() + 1] = Offset;
        __syncthreads();
        if (ThreadId() < warpsize) {
            auto OrphanSum = PrefixStarts[ThreadId() + 1];
            #pragma unroll
            for (auto i = 1; i < WarpCount; i *= 2) {
                const auto Add = __shfl_up_sync(Everyone, OrphanSum, i);
                if (ThreadId() >= i) { OrphanSum += Add; }
            }
            //if (ThreadId() < WarpCount) { PrefixStarts[ThreadId() + 1] = (OrphanSum + StartCount); }
            PrefixStarts[ThreadId() + 1] = (OrphanSum + StartCount);
        }
        __syncthreads();
        return PrefixStarts[WarpCount]; //start point for the next iteration
    };


    __syncthreads();
    //1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 
    //Phase 1. The redzone contains the source items that will be used to fill the holes
    //The removals contain the destination items that will be removed
    //If a redzone item is itself to be removed, then it is not a valid source item
    //Mark all those items as invalid.
    printonceif(DebugPrint, "Start Phase 1: mark all removal items inside the redzone\n");
    printonceif(DebugPrint, "RedZoneStart = %i, FalseCount = %i, removeCount = %i\n", RedZoneStart, Length, RemoveCount);
    
    for (auto i = DeviceThreadId(); i < RemoveCount; i += gridsize) {
        const auto dest = Removals[i];
        if (dest >= RedZoneStart) {
            printif(DebugPrint, "Item[%i]:$%x(%i) Source[%i] = removed\n", i, dest, dest, dest - RedZoneStart);
            assert2((Removals[dest - RedZoneStart] & InvalidSource) == 0, "Oops, trying to remove an item[%i] that is already removed", dest - RedZoneStart);
            //Removals[dest - RedZoneStart] |= InvalidSource;
            List[dest] = InvalidSource;
        }
    } // for
    g.sync();
    if (DeviceThreadId() == 0) {
        const auto Phase1Time = clock64();
        BlockStore.PhaseTimes[0] = Phase1Time - BlockStore.PhaseTimes[0];
        BlockStore.PhaseTimes[1] = Phase1Time;
    }

    auto ItemsPerBlock = DivUp<BlockCount>(RemoveCount);
    const auto BlockStart = BlockId() * ItemsPerBlock;
    ItemsPerBlock = std::min(ItemsPerBlock, RemoveCount - BlockStart);



    const auto Prefetch1 = [&](const int run) {
        pipeline.producer_acquire();
        const auto Offset = run * blocksize;
        const auto start = Offset + (ThreadId()); //every block gets its own run, hence the use of ThreadId() and not DeviceThreadId()
        const int size = std::max(std::min((ItemsPerBlock - start), 1), 0);
        if (size) {
            const auto r = mod<prefetchcount>(run);
            memcpy_async(&ListBuffer[ThreadId() + (r * blocksize)], &List[RedZoneStart + BlockStart + start], sizeof(T), pipeline);
            memcpy_async(&RemoveBuffer[ThreadId() + (r * blocksize)], &Removals[BlockStart + start], sizeof(int), pipeline);
        }
        pipeline.producer_commit();
    };

    //DoPrefetch(0);
    #pragma unroll
    for (auto i = 0; i < prefetchcount; i++) { Prefetch1(i); }

    printonceif(DebugPrint, "Start Phase 2, Move valid sources to valid dest, save invalid sources/dest. in the orphan list");
    //2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
    //Phase 2. We have sources in the redzone (Z), and destination (indices) in r
    //both the source and the destination can be either valid or invalid
    //This means we have 4 cases to deal with
    // 1. invalid source Z, invalid destination r -> do nothing
    // 2. valid source, invalid destination       -> store the source in the orphan list
    // 3. invalid source, valid destination       -> store the destination in the correction list
    // 4. valid source, valid destination         -> move the source to the destination  
    //Every block saves orphans in its own part of Removals(R) and RedZone(Z)
    //By doing so we do not run into racing conditions where we try to write, but some other block has not read yet.

    auto SourceOrphanCount = 0;
    auto DestOrphanCount = 0;
    const auto MaxRuns = DivUp<blocksize>(ItemsPerBlock);
    for (auto run = 0, i = ThreadId(); run < MaxRuns; run++, i += blocksize) {  //we need all threads active to create the prefix sum

        //we only need to mark out of range items on the last iteration, can we lift this out of the loop?
        auto dest = uint32_t(-1);
        T source = InvalidSource;

        cuda::pipeline_consumer_wait_prior<prefetchcount - 1>(pipeline);

        if (i < ItemsPerBlock) {
            dest = RemoveBuffer[mod<blocksize * prefetchcount>(i)];
            source = ListBuffer[mod<blocksize * prefetchcount>(i)];       //the item to fill the hole with
            #ifdef _DEBUG
            const decltype(dest) solldest = Removals[BlockStart + i];
            const decltype(source) sollsource = List[RedZoneStart + BlockStart + i];
            if ((solldest != dest) || (source != sollsource)) {
                assert2(solldest == dest, "run = %i, BlockId = %i, i = %i, dest = %x, solldest = %x", run, BlockId(), i, dest, solldest);
                assert2(sollsource == source, "run = %i, BlockId = %i, i = %i, source = %x, sollsource = %x", run, BlockId(), i, source, sollsource);
            }
            #endif
            printif(DebugPrint, "************ in the redzone ************ i= %i, removeIndex(dest) = %i, isDeleted(dest) = %i\n", i + BlockStart, DestIndex(dest), !GetIsSourceValid(dest));
            printif(DebugPrint, "Source: List[RedZoneStart(%i) + %i=%i] = %i\n", RedZoneStart, BlockStart + i, RedZoneStart + BlockStart + i, source);
        }

        Prefetch1(run + prefetchcount);

        const auto isDestValid = (DestIndex(dest) < RedZoneStart);
        const auto isSourceValid = InvalidSource != source; //GetIsSourceValid(dest);
        const auto isSourceOrphan = (isSourceValid && (!isDestValid));   //source is valid, but destination is invalid -> store the source as orphan
        const auto SourceOrphanMask = __ballot_sync(Everyone, isSourceOrphan);
        SourceOrphanCount = PrefixOffsets(PrefixStartsSource, __popc(SourceOrphanMask), SourceOrphanCount);
        const auto SourceOrphanIndex = LanesBefore(SourceOrphanMask) + PrefixStartsSource[WarpId()];

        if (isSourceOrphan) {
            assert(!isDestValid && isSourceValid);
            List[RedZoneStart + BlockStart + SourceOrphanIndex] = source;
            assert1(__popc(__match_any_sync(SourceOrphanMask, SourceOrphanIndex)) == 1); //do not allow duplicate indices
            printif(DebugPrint, "SourceOrphans[ %i + %i ] = Source(%i)\n", RedZoneStart + BlockStart, SourceOrphanIndex, source);
        }

        //If the destination is valid, but the source is not, then store the dest as orphan
        const auto isDestOrphan = ((!isSourceValid) && isDestValid);
        const auto DestOrphanMask = __ballot_sync(Everyone, isDestOrphan);
        DestOrphanCount = PrefixOffsets(PrefixStartsDest, __popc(DestOrphanMask), DestOrphanCount);
        const auto DestOrphanIndex = LanesBefore<lt>(DestOrphanMask) + PrefixStartsDest[WarpId()];

        if (isDestOrphan) {
            //We need to store these items, because some of them might reference items that will get orphaned in future iterations
            assert1((dest & InvalidSource) == InvalidSource);
            assert1(!isSourceOrphan && !isSourceValid);
            assert1(__popc(__match_any_sync(DestOrphanMask, DestOrphanIndex)) == 1); //do not allow duplicate indices
            assert1(CleanDest(dest) < RedZoneStart);
            Removals[BlockStart + DestOrphanIndex] = dest;//CleanDest(dest);
            printif(DebugPrint, "DestOrphans[ %i + %i ] = %i\n", BlockStart + DestOrphanIndex, DestOrphanIndex, CleanDest(dest));
        }

        const auto isAllowed = (isSourceValid && isDestValid);
        if (isAllowed) {
            assert1(__popc(__match_any_sync(__activemask(), DestIndex(dest))) == 1); //do not allow duplicate dest
            assert1(dest < RedZoneStart);

            //__stcs(&List[dest], source); //no need to strip Removed from source.
            List[dest] = source; //no need to strip Removed from source.
            printif(DebugPrint, "Dest[%i] <= %i\n", dest, source);

        }
        #ifdef _DEBUG
        if (!isSourceValid && !isDestValid) { assert1((!isAllowed) && (!isSourceOrphan) && (!isDestOrphan)); }
        #endif


    } //for phase 2
    __syncthreads();

    //BlockStore.Wait();

    //3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 
    //All the normal items have been processed, now we need to process the orphaned items
    //Phase 3, pair up all the corrections with the orphans
    printonceif(DebugPrint, "************ Entering phase 3: pair up SourceOrphans with DestOrphans ************\n");
    //assert2(isEqual(DestOrphanCount, SourceOrphanCount), "DestOrphanCount = %i, SourceOrphanCount = %i, D-S = %i", DestOrphanCount, SourceOrphanCount, (DestOrphanCount - SourceOrphanCount));

    //Create a presum of all the DestOrphanCounts and SourceOrphanCounts
    __shared__ int DestCounts[BlockCount * 2 + 1];
    __shared__ int SourceCounts[BlockCount * 2 + 1];
    __shared__ int DestTotals[DivUp<warpsize>(BlockCount) + 1];
    __shared__ int SourceTotals[DivUp<warpsize>(BlockCount) + 1];

    if (ThreadId() == 0) {
        BlockStore.DestCounts[BlockId()] = DestOrphanCount;
        BlockStore.SourceCounts[BlockId()] = SourceOrphanCount;
        DestCounts[0] = 0;
        SourceCounts[0] = 0;
    }
    //BlockStore.Wait();
    g.sync();
    if (DeviceThreadId() == 0) {
        const auto Phase2Time = clock64();
        BlockStore.PhaseTimes[1] = Phase2Time - BlockStore.PhaseTimes[1];
        BlockStore.PhaseTimes[2] = Phase2Time;
    }
    if (ThreadId() < BlockCount) {
        DestCounts[ThreadId() + 1] = BlockStore.DestCounts[ThreadId()];
        SourceCounts[ThreadId() + 1] = BlockStore.SourceCounts[ThreadId()];
    }
    printonceif(DebugPrint, "DestCounts = [%i, %i, %i, %i]\n", DestCounts[0], DestCounts[1], DestCounts[2], DestCounts[3]);
    printonceif(DebugPrint, "SourceCounts = [%i, %i, %i, %i]\n", SourceCounts[0], SourceCounts[1], SourceCounts[2], SourceCounts[3]);

    const auto PrefixCounts = [](int* Counts, int* Totals) {
        __syncthreads();
        for (auto w = WarpId(); w < DivUp<warpsize>(BlockCount); w += WarpCount) {
            auto Sum = 0;
            const auto tid = LaneId() + (w * warpsize);
            if (tid <= BlockCount) { Sum = Counts[tid]; }
            for (auto i = 1; i < warpsize; i *= 2) {
                const auto Add = __shfl_up_sync(Everyone, Sum, i);
                if (LaneId() >= i) { Sum += Add; }
            }
            if (tid <= BlockCount) { Counts[tid] = Sum; }
        }
        __syncthreads();
        //Collect the last value from each warp, do a prefix sum on that and add it to the warp totals
        //This limits the number of blocks to 32 * 32 = 1024
        if constexpr (BlockCount >= warpsize) {
            constexpr auto WorkCount = DivUp<warpsize>(BlockCount);
            if (WarpId() == 0) {
                //printonce("@@@@@@@@@@@@@@@@");
                auto Sum = 0;
                const auto ActiveMask = __ballot_sync(Everyone, ThreadId() < WorkCount);
                if (ThreadId() < WorkCount) {
                    Sum = Counts[(ThreadId() * warpsize) + (warpsize - 1)];

                    for (auto i = 1; i < WorkCount; i *= 2) {
                        const auto Add = __shfl_up_sync(ActiveMask, Sum, i);
                        if (LaneId() >= i) { Sum += Add; }
                    }
                    Totals[LaneId()] = Sum;
                }
            }
            __syncthreads();
            for (auto w = WarpId(); w < WorkCount; w += WarpCount) {
                const auto dest = (w + 1) * warpsize + LaneId();
                if (dest <= BlockCount) { Counts[dest] += Totals[w]; }
            }
            if (ThreadId() == 0) { Counts[BlockCount + 1] = Counts[BlockCount]; }
            __syncthreads();
        }
    };
    PrefixCounts(DestCounts, DestTotals);
    PrefixCounts(SourceCounts, SourceTotals);
    printif(DebugPrint, "DestCounts = [%i, %i, %i, %i]\n", DestCounts[0], DestCounts[1], DestCounts[2], DestCounts[3]);
    printif(DebugPrint, "SourceCounts = [%i, %i, %i, %i]\n", SourceCounts[0], SourceCounts[1], SourceCounts[2], SourceCounts[3]);
    //Do a load balance on the orphan data.
    const auto OrphanCount = DestCounts[BlockCount];
    assert2(OrphanCount == SourceCounts[BlockCount], "DestOrphanCount = %i, SourceOrphanCount = %i", SourceCounts[BlockCount], DestCounts[BlockCount]);
    #ifdef _DEBUG
    for (auto i = 0; i <= (BlockCount * 2 + 2); i++) {
        printif(DeviceThreadId() == 0, "DestCounts[%i] = %i, SourceCounts[%i] = %i\n", i, DestCounts[i], i, SourceCounts[i]);
    }
    #endif
    assert1((0 == SourceCounts[0]) && (0 == DestCounts[0]));
    g.sync();
    const auto MaxRuns3 = DivUp<gridsize>(OrphanCount);
    const auto BlockOffset = DivUp<BlockCount>(RemoveCount);
    for (auto run = 0; run < MaxRuns3; run++) {
        const auto tid = DeviceThreadId() + (run * gridsize);
        const auto DestOrphanIndex = LoadBalance<WarpCount, BlockCount>(run, DestCounts);
        const auto SourceOrphanIndex = LoadBalance<WarpCount, BlockCount>(run, SourceCounts);

        if (-1 == DestOrphanIndex) {
            assert1(tid >= OrphanCount);
            assert(DeviceThreadId() > 0);
            assert1(-1 == SourceOrphanIndex);
        } else {
            const auto DestSubIndex = tid - DestCounts[DestOrphanIndex];
            const auto SourceSubIndex = tid - SourceCounts[SourceOrphanIndex];
            printif(DebugPrint, "tid = %i, DestOrphanIndex = %i %i, SourceOrphanIndex = %i %i\n", tid, DestOrphanIndex, DestSubIndex, SourceOrphanIndex, SourceSubIndex);
            assert1(SourceOrphanIndex >= 0);
            const auto dest = Removals[BlockOffset * DestOrphanIndex + DestSubIndex];
            const auto source = List[RedZoneStart + BlockOffset * SourceOrphanIndex + SourceSubIndex];
            assert2(dest < RedZoneStart, "dest = %i(%x), RedzoneStart = %i, d-R = %i", dest, dest, RedZoneStart, dest - RedZoneStart);
            List[dest] = source;
        }
    }
    g.sync();
    if (DeviceThreadId() == 0) {
        const auto Phase3Time = clock64();
        BlockStore.PhaseTimes[2] = Phase3Time - BlockStore.PhaseTimes[2];
    }
    g.sync();
}

template <int BlockSize, int BlockCount>
__global__ void TestLoadBalancing() {
    constexpr auto gridsize = BlockSize * BlockCount * warpsize;
    __shared__ int SharedCount[10];
    SharedCount[0] = 0;
    SharedCount[1] = 8;
    SharedCount[2] = 10;
    SharedCount[3] = 17;
    SharedCount[4] = 29;
    SharedCount[5] = 100;
    SharedCount[6] = 129;
    SharedCount[7] = 129;
    SharedCount[8] = 129;
    __syncthreads();
    auto index = LoadBalance<BlockSize, BlockCount>(0, &SharedCount[1], 6);
    print("index = %i\n", index);
    index = LoadBalance<BlockSize, BlockCount>(gridsize, &SharedCount[1], 6);
    print("index = %i\n", index);
}

template <int WarpCount, typename T, typename IsValid>
__device__ int CompactBallot(T* data, int length, IsValid isValid) {
    //Read the data into registers
    __shared__ int WarpOffsets[warpsize + 1];
    if (ThreadId() <= WarpCount) { WarpOffsets[ThreadId()] = 0; }
    for (auto i = ThreadId(); i < length; i += BlockSize()) {
        const auto item = data[i];
        const auto ValidMask = __ballot_sync(Everyone, isValid(item));
        const auto ValidCount = __popc(ValidMask);
        static_assert(WarpCount <= warpsize); //Max threads per block <= 1024 up to SM 9.x
        static_assert(((WarpCount + (warpsize - 1)) / warpsize) == 1);  //warpcount must be <= warpsize
        WarpOffsets[WarpId() + 1] += ValidCount;
        __syncthreads();
        if (WarpId() < 1) {
            auto prefix = WarpOffsets[ThreadId()];
            #pragma unroll
            for (auto i = 1; i < WarpCount; i *= 2) {
                prefix += __shfl_down_sync(Everyone, prefix, i);
            }
            if (LaneId() < warpsize) {
                WarpOffsets[LaneId() + 1] += prefix;
            }
        }
        __syncthreads();
        auto start = WarpOffsets[WarpId()];
        start += LanesBefore(ValidMask);
        data[start] = item;
        if (WarpId() == 0) {
            const auto end = WarpOffsets[WarpCount];
            WarpOffsets[LaneId()] = end;
        }
        __syncthreads();
    } //for i
    __syncthreads();
    return WarpOffsets[WarpCount];
}

__device__ void FillOriginalData(int* data, int length) {
    for (auto i = ThreadId(); i < length; i += BlockSize()) {
        data[i] = i;
    }
}


__device__ void CopyDataAndMarkRemovals(int* dest, const int* OriginalData, const int* RemoveItems, const int length, const int RemoveLength) {
    for (auto i = ThreadId(); i < length; i += BlockSize()) {
        dest[i] = OriginalData[i];
    }
    __syncthreads();
    for (auto i = ThreadId(); i < RemoveLength; i += BlockSize()) {
        const auto index = RemoveItems[i];
        dest[index] = -1;
    }
    __syncthreads();
}

template <int size_n, int size_k>
__global__ void GlobalCompact(int* data, int* OriginalData, int* RemoveItems, int* Orphans[]) {
    //put data into buffer
    //1. all in shared memory
    //2. K small N big
    //3. K fits into shared memory, N not
    //4. K and N in global memory
    //5. K, N, orphans in global memory

    //1. all in shared memory
    FillOriginalData(OriginalData, size_n);
    GetRemovals<size_k>(RemoveItems);
    __syncthreads();

    //thrust
    CopyDataAndMarkRemovals(data, OriginalData, RemoveItems, size_n, size_k);
    auto TimeThrust = clock64();
    auto length = ThrustCompact(data, size_n, [RemoveItems](const auto& item) { return RemoveItems[item] == -1; });
    TimeThrust = clock64() - TimeThrust;
    const auto ThurstLength = length;


    CopyDataAndMarkRemovals(data, OriginalData, RemoveItems, size_n, size_k);
    auto TimeBallot = clock64();
    switch ((BlockSize() + 31) / 32) {
        case 1: length = CompactBallot<1>(data, size_n, [RemoveItems] __device__(const auto & item) { return item != -1; });    break;
        case 2: length = CompactBallot<2>(data, size_n, [RemoveItems] __device__(const auto & item) { return item != -1; });    break;
        case 3: length = CompactBallot<3>(data, size_n, [RemoveItems] __device__(const auto & item) { return item != -1; });    break;
        case 4: length = CompactBallot<4>(data, size_n, [RemoveItems] __device__(const auto & item) { return item != -1; });    break;
        case 5: length = CompactBallot<5>(data, size_n, [RemoveItems] __device__(const auto & item) { return item != -1; });    break;
        case 6: length = CompactBallot<6>(data, size_n, [RemoveItems] __device__(const auto & item) { return item != -1; });    break;
        case 7: length = CompactBallot<7>(data, size_n, [RemoveItems] __device__(const auto & item) { return item != -1; });    break;
        case 8: length = CompactBallot<8>(data, size_n, [RemoveItems] __device__(const auto & item) { return item != -1; });    break;
        case 9: length = CompactBallot<9>(data, size_n, [RemoveItems] __device__(const auto & item) { return item != -1; });    break;
        case 10: length = CompactBallot<10>(data, size_n, [RemoveItems] __device__(const auto & item) { return item != -1; });  break;
        case 11: length = CompactBallot<11>(data, size_n, [RemoveItems] __device__(const auto & item) { return item != -1; });  break;
        case 12: length = CompactBallot<12>(data, size_n, [RemoveItems] __device__(const auto & item) { return item != -1; });  break;
        case 13: length = CompactBallot<13>(data, size_n, [RemoveItems] __device__(const auto & item) { return item != -1; });  break;
        case 14: length = CompactBallot<14>(data, size_n, [RemoveItems] __device__(const auto & item) { return item != -1; });  break;
        case 15: length = CompactBallot<15>(data, size_n, [RemoveItems] __device__(const auto & item) { return item != -1; });  break;
        case 16: length = CompactBallot<16>(data, size_n, [RemoveItems] __device__(const auto & item) { return item != -1; });  break;
        case 17: length = CompactBallot<17>(data, size_n, [RemoveItems] __device__(const auto & item) { return item != -1; });  break;
        case 18: length = CompactBallot<18>(data, size_n, [RemoveItems] __device__(const auto & item) { return item != -1; });  break;
        case 19: length = CompactBallot<19>(data, size_n, [RemoveItems] __device__(const auto & item) { return item != -1; });  break;
        case 20: length = CompactBallot<20>(data, size_n, [RemoveItems] __device__(const auto & item) { return item != -1; });  break;
        case 21: length = CompactBallot<21>(data, size_n, [RemoveItems] __device__(const auto & item) { return item != -1; });  break;
        case 22: length = CompactBallot<22>(data, size_n, [RemoveItems] __device__(const auto & item) { return item != -1; });  break;
        case 23: length = CompactBallot<23>(data, size_n, [RemoveItems] __device__(const auto & item) { return item != -1; });  break;
        case 24: length = CompactBallot<24>(data, size_n, [RemoveItems] __device__(const auto & item) { return item != -1; });  break;
        case 25: length = CompactBallot<25>(data, size_n, [RemoveItems] __device__(const auto & item) { return item != -1; });  break;
        case 26: length = CompactBallot<26>(data, size_n, [RemoveItems] __device__(const auto & item) { return item != -1; });  break;
        case 27: length = CompactBallot<27>(data, size_n, [RemoveItems] __device__(const auto & item) { return item != -1; });  break;
        case 28: length = CompactBallot<28>(data, size_n, [RemoveItems] __device__(const auto & item) { return item != -1; });  break;
        case 29: length = CompactBallot<29>(data, size_n, [RemoveItems] __device__(const auto & item) { return item != -1; });  break;
        case 30: length = CompactBallot<30>(data, size_n, [RemoveItems] __device__(const auto & item) { return item != -1; });  break;
        case 31: length = CompactBallot<31>(data, size_n, [RemoveItems] __device__(const auto & item) { return item != -1; });  break;
        case 32: length = CompactBallot<32>(data, size_n, [RemoveItems] __device__(const auto & item) { return item != -1; });  break;
    }
    TimeBallot = clock64() - TimeBallot;
    const auto BallotLength = length;

    auto TimeNew = clock64();
    length = CompactNew(data, size_n, [RemoveItems] __device__(const auto & item) { return item != 0; });
    TimeNew = clock64() - TimeNew;
    const auto NewLength = length;
    assert1(BallotLength == NewLength);
    assert1(ThurstLength == NewLength);
    assert1(ThurstLength == (size_n - size_k));
}

template <bool OneBlock>
void test_redzone(const int run) {
    constexpr auto maxelements = 1 << 29;
    TestLists_t TestList(maxelements, maxelements);

    const int rp[] = { 100, -1, 0, 25, 50, 75 };
    BlockStorage_t<GPUBlockCount> BlockStore;
    BlockStorage_t<GPUBlockCount> HostBlockStore;
    float* dev_TestTime;

    CUDA(cudaMalloc(&dev_TestTime, sizeof(float) * 4));
    //printf("Test of redzone GPU wide\n");
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

                TestList.Update(listsize, k, redzone_percentage);
                
                //init the data
                constexpr auto GPUWarpCount = GPUBlockSize / 32;
                CUDA(cudaFuncSetAttribute(TestGPUWide<GPUWarpCount, GPUBlockCount, 5>, cudaFuncAttributeMaxDynamicSharedMemorySize, 48 * 1024));
                void* kernelArgs[] = { 
                    &TestList.dev_A, &TestList.dev_Asize, 
                    &TestList.dev_R, &TestList.dev_Rsize, 
                    &BlockStore.self, &TestList.dev_TestTime };
                dim3 dimBlock(GPUBlockSize, 1, 1);
                dim3 dimGrid(GPUBlockCount, 1, 1);
                constexpr auto prefetchcount = 5;
                const auto SharedMemSize = 1024 * sizeof(int) * 2 * prefetchcount;
                int MaxAllowedBlocks;
                CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&MaxAllowedBlocks, TestGPUWide<DivUp<32>(GPUBlockSize), GPUBlockCount, 5>, GPUBlockSize, SharedMemSize, cudaOccupancyDefault));
                //printf("MaxAllowedBlocks = %i\n", MaxAllowedBlocks);
                const auto StartCPUTime = std::chrono::high_resolution_clock::now();
                if constexpr (OneBlock) {
                    //printf("dev_A = %p, Asize = %i, dev_R = %p, Rsize = %i, devTime = %p\n", TestList.dev_A, TestList.Asize, TestList.dev_R, TestList.Rsize, TestList.dev_TestTime);
                    RedzoneOneBlock<DivUp<32>(GPUBlockSize), false><<<1, GPUBlockSize>>>(TestList.dev_A, TestList.Asize, TestList.dev_R, TestList.Rsize, TestList.dev_TestTime);
                } else {
                    //all available blocks, using the cooperative groups
                    CUDA(cudaLaunchCooperativeKernel((void*)TestGPUWide<DivUp<32>(GPUBlockSize), GPUBlockCount, 5>, dimGrid, dimBlock, kernelArgs, SharedMemSize));
                }
                CUDA(cudaDeviceSynchronize());
                const auto StopCPUTime = std::chrono::high_resolution_clock::now();
                const auto CPUTime = (int)std::chrono::duration_cast<std::chrono::microseconds>(StopCPUTime - StartCPUTime).count();
                float Times[4];
                CUDA(cudaMemcpy(&Times[0], TestList.dev_TestTime, sizeof(float) * 4, cudaMemcpyDeviceToHost));
                //for short runs this will include the >= 12-25us kernel startup time
                //and will arbitrarily assign this to the phases, but that's fine
                const auto TotalGPUTime = Times[1] + Times[2] + Times[3];
                const auto divisor = (0.0f == CPUTime || 0.0f == TotalGPUTime) ? 1.0f : TotalGPUTime / CPUTime;
                const auto time1 = int(Times[1] / divisor);
                const auto time2 = int(Times[2] / divisor);
                const auto time3 = int(Times[3] / divisor);
                const auto time0 = int(Times[0] / 1.815f); //GPU frequency
                const auto is_OK = TestList.isOK();
                const auto status = is_OK ? "Pass" : "##### error!!!!!";
                printf("n = 1 << %i, k = %i%%, r = %i%%, time1 = %i, time2 = %i, time3 = %i, timeall = %i, cputime = %i, d = %.2f, OK = %s, run = %i\n", e, kpercentage, redzone_percentage, time1, time2, time3, time0, CPUTime, divisor, status, run);
            }
        }
    }
}

void test_ON_OneBlock(const int run) {
    constexpr auto maxelements = 1 << 29;
    TestLists_t TestList(maxelements, maxelements);

    auto errors = 0;
    int kp[] = { 2, 10, 50, 90 };
    //data elements from 2^5 to 2^29
    for (auto r = 0; r < 4; r++) {
        for (auto e = 29; e >= 14; e--) {
            const auto NELEMENTS = 1 << e;
            constexpr auto blockSize = GPUBlockSize;
            const auto k = (uint64_t(NELEMENTS) * kp[r]) / 100;
            TestList.Update(NELEMENTS, k);
            
            const auto datasize = sizeof(int) * NELEMENTS;
            //host input/output data
            auto d_data = TestList.dev_A;
            auto d_output = TestList.dev_A2;
            auto h_data = TestList.host_A;
            const auto start1 = std::chrono::high_resolution_clock::now();
            constexpr auto WarpCount = DivUp<32>(blockSize);
            TestO_N_Removal_OneBlock<WarpCount, false, true> << <1, blockSize >> > (d_data, NELEMENTS, TestList.dev_R, TestList.Rsize, TestList.dev_TestTime);
            CUDA(cudaDeviceSynchronize());
            const auto end1 = std::chrono::high_resolution_clock::now();
            const auto t1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count();

            //copy back results to host
            CUDA(cudaMemcpy(h_data, d_output, datasize, cudaMemcpyDeviceToHost));
            float devTimes[4];
            CUDA(cudaMemcpy(devTimes, TestList.dev_TestTime, sizeof(float) * 4, cudaMemcpyDeviceToHost));
            float Correction = t1 / devTimes[0];
            devTimes[1] *= Correction;
            devTimes[2] *= Correction;
            //printData(h_data,NELEMENTS);
            const auto is_OK = TestList.isOK();
            errors += (!is_OK);
            printf("algo='OnePass', n = %i, k = %i%%, t1 = %.2f, t2 = %.2f, tt = %i, %s, run = %i\n", e, kp[r], devTimes[1], devTimes[2], t1, is_OK ? "Pass" : "Fail", run);
        } //for blocksize
    } //for elements
}