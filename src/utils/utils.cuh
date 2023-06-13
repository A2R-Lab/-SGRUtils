/*
    Cuda utility functions
*/
#include <cuda_runtime.h>
#include <stdio.h>
// #include <iiwa-grid.cuh>

template <typename T>
__device__ T* shared_memory_proxy()
{
    // __align__(sizeof(T)) -- this will break if multiple Ts chosen
    extern __shared__ unsigned char memory[];
    return reinterpret_cast<T*>(memory);
}

__device__ void sleepKernel()
{
    unsigned long long int start = clock64(); // get the start time
    unsigned long long int delay = 1000000000; // sleep for 1 second
    
    while (clock64() < start + delay) {
        // busy-wait
    }
    
    printf("Thread %d woke up after sleeping for 1 second.\n", threadIdx.x);
}

#define gpuErrchk(ans) { sgrutils::gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template <typename T>
__device__
__host__
void printMatrixColumnMajor(T * matrix, int rows, int cols) {
    for(int i=0; i<rows; i++) {
        for(int j=0; j<cols; j++) {
            printf("%f ", matrix[j*rows + i]);
        }
        printf("\n");
    }
}

template <typename T>
__device__
__host__
void printMatrix(T * matrix, int rows, int cols) {
    for(int i=0; i<rows; i++) {
        for(int j=0; j<cols; j++) {
            printf("%f ", matrix[i*cols + j]);
        }
        printf("\n");
    }
}

// copies a matrix (optionally scales by alpha)
template <typename T, int M, int N>
__host__ __device__ __forceinline__
void copyMat(T *dst, T *src, int ld_dst, int ld_src, T alpha = 1.0){
    int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
    #pragma unroll
    for (int ky = starty; ky < N; ky += dy){
        #pragma unroll
        for (int kx = startx; kx < M; kx += dx){
            dst[kx + ld_dst*ky] = alpha*src[kx + ld_src*ky];
        }
    }
}

// loads a matrix into shared memory
// special case of copyMat, assumes shared memory is of size m*n and original matrix is on disk size ld*n
template <typename T, int M, int N>
__host__  __device__ __forceinline__
void loadMatToShared(T *dst, T *src, int ld){
    copyMat<T,M,N>(dst, src, M, ld);
}

// loads the identiy matrix into a variable
template <typename T, int M, int N>
__host__  __device__ __forceinline__
void loadIdentity(T *A, int ld_A){
    int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
    #pragma unroll
    for (int ky = starty; ky < N; ky += dy){
        #pragma unroll
        for (int kx = startx; kx < M; kx += dx){
            A[ky*ld_A + kx] = static_cast<T>(kx == ky ? 1 : 0);
        }
    }
}