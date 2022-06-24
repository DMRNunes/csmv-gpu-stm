#ifndef _CUDA_UTIL_H_GUARD
#define _CUDA_UTIL_H_GUARD

#include <numa.h>

// #define LOG_POW2_BEFORE(value) ({
//   double new_exp__ = log((double)value) / log(2.0D); \
//   llround((double)pow(2.0D, (size_t)new_exp__)); \
// })

#define LOG_MOD2(idx, mod) ({ \
  /* assert(__builtin_popcount((mod)) == 1); */ /* base 2 mod */ \
  (long long)(idx) & (((unsigned long long)mod) - 1); \
})

// #define LOG_DISTANCE2(start, end, mod) ({ \
//   LOG_MOD2(((long long)(end) - (long long)(start)), (mod)); \
// })

#define T_ERROR(cond) ({ \
	if (!(cond)) { \
		fprintf(stderr, #cond ": in " __FILE__ ":%i \n   > %s\n", \
		__LINE__, strerror(errno)); \
	} \
})

#define CUDA_CHECK_ERROR(func, msg) ({ \
	cudaError_t cudaError; \
	if (cudaSuccess != (cudaError = func)) { \
		fprintf(stderr, #func ": in " __FILE__ ":%i : " msg "\n   > %s\n", \
		__LINE__, cudaGetErrorString(cudaError)); \
    *((int*)0x0) = 0; /* exit(-1); */ \
	} \
  cudaError; \
})

#define CUDA_DEV_ALLOC(ptr, size) \
	CUDA_CHECK_ERROR(cudaMalloc((void**)&(ptr), size), \
		"[cudaMalloc]: failed for " #ptr);

#define CUDA_HOST_ALLOC(ptr, size) \
	CUDA_CHECK_ERROR(cudaMallocHost((void**)&(ptr), size), \
		"[cudaMallocHost]: failed for " #ptr);

#define CUDA_DUAL_ALLOC(ptrHost, ptrDev, size) \
	CUDA_HOST_ALLOC(ptrHost, size); \
	CUDA_DEV_ALLOC(ptrDev, size) \

// #define CUDA_HOST_ALLOC(ptr, size) \
// 	ptr = (__typeof__(ptr))numa_alloc_interleaved(size);

#define CUDA_CPY_TO_DEV(dev, host, size) \
	CUDA_CHECK_ERROR(cudaMemcpy((void*)dev, (void*)host, size, \
	cudaMemcpyHostToDevice), "[cudaMemcpyHostToDevice]: failed for " \
		#host " --> " #dev)

#define CUDA_CPY_TO_DEV_ASYNC(dev, host, size, stream) \
	CUDA_CHECK_ERROR(cudaMemcpyAsync((void*)dev, (void*)host, size, \
	cudaMemcpyHostToDevice, stream), "[cudaMemcpyHostToDevice]: failed for " \
		#host " --> " #dev)

#define CUDA_CPY_TO_HOST(host, dev, size) \
	CUDA_CHECK_ERROR(cudaMemcpy((void*)host, (void*)dev, size, \
	cudaMemcpyDeviceToHost), "[cudaMemcpyDeviceToHost]: failed for " \
		#dev " --> " #host)

#define CUDA_CPY_TO_HOST_ASYNC(host, dev, size, stream) \
	CUDA_CHECK_ERROR(cudaMemcpyAsync((void*)host, (void*)dev, size, \
	cudaMemcpyDeviceToHost, stream), "[cudaMemcpyDeviceToHost]: failed for " \
		#dev " --> " #host)

#define CUDA_CPY_DtD_ASYNC(dev1, dev2, size, stream) \
	CUDA_CHECK_ERROR(cudaMemcpyAsync((void*)dev1, (void*)dev2, size, \
	cudaMemcpyDeviceToDevice, stream), "[cudaMemcpyDeviceToDevice]: failed for " \
		#dev2 " --> " #dev1)

#endif /* _CUDA_UTIL_H_GUARD */
