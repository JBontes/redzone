# Redzone algorithm
Redzone: fast stream compaction: removing k items from a list in parallel O(k) time

Stream compaction is the parallel removal of *k* items from a list with *n* total items. Current parallel deletion algorithm use parition which takes O(n) time.
However, sequential algorithms are often faster because they take O(*k*) time. Redzone is a parallel algorithm that removes items in O(*k*) time. 
It has low constant overhead as is typically faster than O(n) parallel stream compaction when *k* < 1/2*n*, and orders of magnitude faster if _k_ << _n_.

The parallel deletion fills holes created by removals with non-deleted items from the tail. In the same way that sequential algorithms typically work. 
It performs some extra work to prevent data races. If an item is not deleted and not part of the tail items used to fill holes, it will not move them, which
may be beneficial if links (e.g.: pointers) exist that reference your items. 
Because Redzone only moves items that need to be moved to keep the list contiguous, it does no more work than necessary.

Drawbacks of Redzone are that it is unstable, i.e.: it will reorder the keep elements in your list; and it needs a list of items to delete (the removal list 
cannot contain duplicates).

For more details on the algorithm, its performance, and a proof of correctness, see the paper: [`Redzone paper.pdf`](https://github.com/JBontes/redzone/blob/main/RedZone%20paper.pdf).

# Compile and run

A version of the algorithm for both CPUs and GPUs is included. The CPU code uses C++20 and has been tested on both Windows (MSVC) and Linux (Clang). 
The GPU code uses [NVCC 12 (from the CUDA 12 toolkit)](https://developer.nvidia.com/cuda-downloads) and runs both on Windows and Linux. Because the demo code uses asynchronous memory transfer it requires compute capacility 70 (Volta) or higher.

To compile the full test you'll need to install [Intel's ~~tbb~~ oneAPI library](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onetbb-download.html)

## Compilation
```
//debug
nvcc -std c++20 -arch=native --extended-lambda --expt-relaxed-constexpr -I"include_path_for_tbb" -L"library_path_for_tbb" tests.cu
//release
nvcc -O3 -g -DNDEBUG -std c++20 -arch=native --extended-lambda --expt-relaxed-constexpr -I"include_path_for_tbb" -L"include_path_for_tbb" tests.cu -o tests2
//On my Windows system
//include_path_for_tbb = "C:\Program Files (x86)\Intel\oneAPI\tbb\2021.12\include"
//library_path_for_tbb = "C:\Program Files (x86)\Intel\oneAPI\tbb\2021.12\lib"
//
```

Make sure that `tbb12_debug.dll` and `tbb12.dll` are in the path, so that the application can find them.

# Citation

If you use this algorithm in your research, please cite it as: 

```
@article{10.1145/3675782,
author = {Bontes, Johan and Gain, James},
title = {Redzone stream compaction: removing k items from a list in parallel O(k) time},
year = {2024},
issue_date = {September 2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {11},
number = {3},
issn = {2329-4949},
url = {https://doi.org/10.1145/3675782},
doi = {10.1145/3675782},
abstract = {Stream compaction, the parallel removal of selected items from a list, is a fundamental building block in parallel algorithms. It is extensively used, both in computer graphics, for shading, collision detection, and ray tracing, as well as in general computing, such as for tree traversal and database selection.In this article, we present Redzone stream compaction, the first parallel stream compaction algorithm to remove k items from a list with n ≥ k elements in O(k) rather than O(n) time. Based on our benchmark experiments on both GPU and CPU, if k is proportionally small (k ≪ n), Redzone outperforms existing parallel stream compaction by orders of magnitude, while if k is close to n, it underperforms by a constant factor. Redzone removes items in-place and needs only O(1) auxiliary space. However, unlike current O(n) algorithms, it is unstable (i.e., the order of elements is not preserved) and it needs a list of the items to be removed.},
journal = {ACM Trans. Parallel Comput.},
month = {aug},
articleno = {14},
numpages = {16},
keywords = {Stream compaction, list removal}
}

  
```
