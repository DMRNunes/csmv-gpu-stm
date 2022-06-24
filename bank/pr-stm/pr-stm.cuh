/**
 * TODO: PR-STM License
 *
 * For performance reasons all functions on PR-STM are implemented
 * in headers
 *
 * TODO: cite PR-STM paper [EuroPar15]
 */
#ifndef PR_STM_H_GUARD
#define PR_STM_H_GUARD

#ifdef __cplusplus // only works on c++

#include <cuda_runtime.h>
#include "cuda_util.h"

#ifndef PR_LOCK_TABLE_SIZE
// must be power 2
#define PR_LOCK_TABLE_SIZE  0x800000
#endif

#ifndef PR_GRANULE_T
#define PR_GRANULE_T        int
#define PR_LOCK_GRANULARITY 4 /* size in bytes */
#define PR_LOCK_GRAN_BITS   2 /* log2 of the size in bytes */
#endif

#ifndef PR_MAX_RWSET_SIZE
#define PR_MAX_RWSET_SIZE   0x2
#endif

// PR-STM extensions, in order to extend PR-STM implement the following MACROs
// ##########################################

// Use PR_BEFORE_RUN_EXT to init the struct in pr_args->pr_args_ext
#ifndef   PR_ARGS_S_EXT
#define   PR_ARGS_S_EXT /* empty */
#endif /* PR_ARGS_S_EXT */

// Use PR_ALLOC to allocate the struct in PR_AFTER_PREPARE_EXT
// and PR_CPY_TO_DEV in PR_BEFORE_RUN_EXT to setup in the GPU.
// Then in PR_AFTER_RUN_EXT use PR_CPY_TO_HOST to retrieve
// the results from the GPU.
#ifndef   PR_DEV_BUFF_S_EXT
#define   PR_DEV_BUFF_S_EXT /* empty */
#endif /* PR_DEV_BUFF_S_EXT */

#ifndef   PR_BEFORE_BEGIN_EXT
#define   PR_BEFORE_BEGIN_EXT(args, pr_args) /* empty */
#endif /* PR_BEFORE_START_EXT */

#ifndef   PR_AFTER_COMMIT_EXT
#define   PR_AFTER_COMMIT_EXT(args, pr_args) /* empty */
#endif /* PR_AFTER_COMMIT_EXT */

#ifndef   PR_BEFORE_KERNEL_EXT
#define   PR_BEFORE_KERNEL_EXT(args, pr_args) /* empty */
#endif /* PR_BEFORE_KERNEL_EXT */

#ifndef   PR_AFTER_KERNEL_EXT
#define   PR_AFTER_KERNEL_EXT(args, pr_args) /* empty */
#endif /* PR_AFTER_KERNEL_EXT */

#ifndef   PR_AFTER_VAL_LOCKS_EXT
/* use args->wset/args->rset to access the write/read sets */
#define   PR_AFTER_VAL_LOCKS_EXT(args) /* empty */
#endif /* PR_AFTER_VAL_LOCKS_EXT */

#ifndef   PR_AFTER_WRITEBACK_EXT
/* i is the idx of the current granule */
#define   PR_AFTER_WRITEBACK_EXT(args, i, addr, val) /* empty */
#endif /* PR_AFTER_WRITEBACK_EXT */

// The following MACROs are called on the kernel launching process
#ifndef   PR_BEFORE_RUN_EXT
#define   PR_BEFORE_RUN_EXT(args) /* empty */
#endif /* PR_BEFORE_RUN_EXT */

#ifndef   PR_AFTER_RUN_EXT
#define   PR_AFTER_RUN_EXT(args) /* empty */
#endif /* PR_AFTER_RUN_EXT */
// ##########################################

// The application must "PR_globalVars;" somewhere in the main file
static const int PR_MAX_NB_STREAMS = 8;
extern int PR_blockNum;
extern int PR_threadNum;
extern int PR_isStart[PR_MAX_NB_STREAMS];
extern int PR_isDone[PR_MAX_NB_STREAMS];
extern long long PR_nbAborts;
extern long long PR_nbCommits;
extern long long PR_nbAbortsStrm [PR_MAX_NB_STREAMS];
extern long long PR_nbCommitsStrm[PR_MAX_NB_STREAMS];
extern long long PR_nbAbortsLastKernel ;
extern long long PR_nbCommitsLastKernel;
extern long long *PR_sumNbAborts ;
extern long long *PR_sumNbCommits;
extern long long PR_nbAbortsSinceCheckpoint ;
extern long long PR_nbCommitsSinceCheckpoint;
extern double PR_kernelTime;
extern cudaStream_t *PR_streams;
extern int PR_currentStream;
extern int PR_streamCount;
extern int *PR_lockTableHost;
extern int *PR_lockTableDev;

extern cudaEvent_t PR_eventKernelStart;
extern cudaEvent_t PR_eventKernelStop;

#define PR_SET(addr, value)       (*((PR_GRANULE_T*)addr) = (PR_GRANULE_T)(value))
#define PR_MOD_ADDR(addr)         ((addr) % (PR_LOCK_TABLE_SIZE))
#define PR_GET_MTX(mtx_tbl, addr) ((mtx_tbl)[PR_MOD_ADDR(((uintptr_t)addr) >> PR_LOCK_GRAN_BITS)])

#define PR_LOCK_NB_LOCK_BITS     2
#define PR_LOCK_NB_OWNER_BITS    22
#define PR_LOCK_VERSION_BITS     24 // PR_LOCK_NB_LOCK_BITS + PR_LOCK_NB_OWNER_BITS
#define PR_LOCK_NB_VERSION_BITS  8 // 32 - PR_LOCK_VERSION_BITS
#define PR_LOCK_VERSION_OVERFLOW 256 // 32 - PR_LOCK_VERSION_BITS

#define PR_GET_VERSION(x)   ( ((x) >> 24) & 0xff    )
#define PR_CHECK_PRELOCK(x) (  (x)        & 0b1     )
#define PR_CHECK_LOCK(x)    ( ((x) >> 1)  & 0b1     )
#define PR_GET_OWNER(x)     ( ((x) >> 2)  & 0x3fffff)
#define PR_MASK_VERSION     0xff000000
#define PR_THREAD_IDX       (threadIdx.x + blockIdx.x * blockDim.x) // TODO: 3D thread-id grid

// TODO: maximum nb threads is 2048*1024
#define PR_PRELOCK_VAL(version, id) ((version << 24) | (id << 2) | 0b01)
#define PR_LOCK_VAL(version, id)    ((version << 24) | (id << 2) | 0b11)

#define PR_CHECK_CUDA_ERROR(func, msg) \
	CUDA_CHECK_ERROR(func, msg) \
//

#define PR_ALLOC(ptr, size) \
	CUDA_DEV_ALLOC(ptr, size) \
//

#define PR_CPY_TO_DEV(dev, host, size) \
	CUDA_CPY_TO_DEV(dev, host, size) \
//

#define PR_CPY_TO_HOST(host, dev, size) \
	CUDA_CPY_TO_HOST(host, dev, size) \
//

#define PR_CPY_TO_HOST_ASYNC(host, dev, size, stream) \
	CUDA_CPY_TO_HOST_ASYNC(host, dev, size, stream) \
//

#define PR_DEVICE       __device__
#define PR_HOST         __host__
#define PR_BOTH         __device__ __host__
#define PR_ENTRY_POINT  __global__

typedef struct PR_rwset_ { // TODO: this should be a sort of a hashmap
	PR_GRANULE_T **addrs;
	PR_GRANULE_T  *values;
	int           *versions;
	size_t         size;
} PR_rwset_s;

typedef struct PR_args_ {
	int     tid;
	void   *pr_args_ext; /* add struct of type PR_ARGS_S_EXT here */
	int    *mtx;
	int     is_abort;
	void   *inBuf;
	size_t  inBuf_size;
	void   *outBuf;
	size_t  outBuf_size;
	int     current_stream;
	// below is private
	PR_rwset_s rset;
	PR_rwset_s wset;
} PR_args_s;

typedef struct pr_buffer_ {
	void  *buf;
	size_t size;
} pr_buffer_s;

// mutex format: TODO: review this
// lock,version,owner in format version<<5|owner<<2|lock

typedef struct pr_tx_args_dev_host_ {
	void   *pr_args_ext; /* add struct of type PR_DEV_BUFF_S_EXT here */
	int    *mtx;
	int    *nbAborts;
	int    *nbCommits;
	void   *inBuf;
	size_t  inBuf_size;
	void   *outBuf;
	size_t  outBuf_size;
	int     current_stream;
} pr_tx_args_dev_host_s;

typedef struct pr_tx_args_ {
	void                 (*callback)(pr_tx_args_dev_host_s);
	pr_tx_args_dev_host_s  dev;
	pr_tx_args_dev_host_s  host;
	void                  *stream;
} pr_tx_args_s;

// This is in pr-stm-internal.cuh (Do not call it!)
#define PR_globalVars \
	int PR_blockNum        = 1; \
	int PR_threadNum       = 1; \
	int PR_isDone[PR_MAX_NB_STREAMS]; /* Wait for kernel finish */ \
	int PR_isStart[PR_MAX_NB_STREAMS]; /* Wait for kernel start */ \
	long long PR_nbAborts  = 0; \
	long long PR_nbCommits = 0; \
	long long PR_nbAbortsStrm [PR_MAX_NB_STREAMS]; \
	long long PR_nbCommitsStrm[PR_MAX_NB_STREAMS]; \
	long long PR_nbAbortsLastKernel  = 0; \
	long long PR_nbCommitsLastKernel = 0; \
	long long *PR_sumNbAborts  = NULL; \
	long long *PR_sumNbCommits = NULL; \
	long long *PR_sumNbAbortsDev  = NULL; \
	long long *PR_sumNbCommitsDev = NULL; \
	long long PR_nbAbortsSinceCheckpoint  = 0; \
	long long PR_nbCommitsSinceCheckpoint = 0; \
	double PR_kernelTime   = 0; \
	int *PR_lockTableHost  = NULL; \
	int *PR_lockTableDev   = NULL; \
	cudaStream_t *PR_streams = NULL; \
	int PR_currentStream     = 0; \
	int PR_streamCount       = 1; /* Number of streams to use */ \
	cudaEvent_t PR_eventKernelStart; \
	cudaEvent_t PR_eventKernelStop; \
//

#define PR_globalKernelArgs \
	pr_tx_args_dev_host_s args

// if a function is called these must be sent (still compatible with STAMP)
#define PR_txCallDefArgs \
	pr_tx_args_dev_host_s &args, PR_args_s &pr_args \
//
#define PR_txCallArgs \
	args, pr_args \
//

// defines local variables
// TODO: we need to have a very good estimate of PR_MAX_RWSET_SIZE
#define PR_allocRWset(set) ({ \
	PR_GRANULE_T *pr_internal_addrs[PR_MAX_RWSET_SIZE]; \
	PR_GRANULE_T pr_internal_values[PR_MAX_RWSET_SIZE]; \
	int pr_internal_versions[PR_MAX_RWSET_SIZE]; \
	set.addrs = pr_internal_addrs; \
	set.values = pr_internal_values; \
	set.versions = pr_internal_versions; \
}) \
//
#define PR_freeRWset(set) /* empty: local variables used */
//

#define PR_enterKernel(_tid) \
	PR_args_s pr_args; \
	PR_allocRWset(pr_args.rset); \
	PR_allocRWset(pr_args.wset); \
	pr_args.tid = _tid; \
	pr_args.mtx = args.mtx; \
	pr_args.inBuf = args.inBuf; \
	pr_args.inBuf_size = args.inBuf_size; \
	pr_args.outBuf = args.outBuf; \
	pr_args.outBuf_size = args.outBuf_size; \
	pr_args.pr_args_ext = args.pr_args_ext; \
	pr_args.current_stream = args.current_stream; \
	PR_beforeKernel_EXT(PR_txCallArgs); \
//
#define PR_exitKernel() \
	PR_freeRWset(pr_args.rset); \
	PR_freeRWset(pr_args.wset); \
	PR_afterKernel_EXT(PR_txCallArgs); \
//

// setjmp is not available --> simple while
#define PR_txBegin() \
do { \
	PR_beforeBegin_EXT(PR_txCallArgs); \
	pr_args.rset.size = 0; \
	pr_args.wset.size = 0; \
	pr_args.is_abort = 0 \
//

#define PR_txCommit() \
	PR_validateKernel(); \
	if (!pr_args.is_abort) { \
		PR_commitKernel(); \
		if (args.nbCommits != NULL) { \
			args.nbCommits[pr_args.tid + blockDim.x*gridDim.x*pr_args.current_stream]++; \
		} \
	} \
	PR_afterCommit_EXT(PR_txCallArgs); \
	if (pr_args.is_abort) { \
		if (args.nbAborts != NULL) { \
			args.nbAborts[pr_args.tid + blockDim.x*gridDim.x*pr_args.current_stream]++; \
		} \
	} \
} while (pr_args.is_abort); \
//

/**
 * Transactional Read.
 *
 * Reads data from global memory to local memory.
 */
#define PR_read(a) ({ \
	PR_GRANULE_T r; \
	r = PR_i_openReadKernel(&pr_args, (PR_GRANULE_T*)(a)); \
	r; \
})

/**
 * Transactional Write.
 *
 * Do calculations and write result to local memory.
 * Version will increase by 1.
 */
#define PR_write(a, v) ({ /*TODO: v cannot be a constant*/ \
	PR_i_openWriteKernel(&pr_args, (PR_GRANULE_T*)(a), v); \
})

/**
 * Validate function.
 *
 * Try to lock all memory this thread need to write and
 * check if any memory this thread read is changed.
 */
#define PR_validateKernel() \
	PR_i_validateKernel(&pr_args)

/**
 * Commit function.
 *
 * Copies results (both value and version) from local memory
 * (write set) to global memory (data and lock).
 */
#define PR_commitKernel() \
	PR_i_commitKernel(&pr_args)

// Use these MACROs to wait execution (don't forget to put a lfence)
#define PR_WAIT_START_COND  (!PR_isStart[PR_currentStream] && !PR_isDone[PR_currentStream])
#define PR_WAIT_FINISH_COND (!PR_isDone[PR_currentStream])

// IMPORTANT: do not forget calling this after wait
#define PR_AFTER_WAIT(args) ({ \
	float kernelElapsedTime; \
	PR_isDone[PR_currentStream] = 0; \
	PR_isStart[PR_currentStream] = 0; \
	/* cudaEventSynchronize(PR_eventKernelStop); done by cudaStreamSynchronize */ \
	cudaEventElapsedTime(&kernelElapsedTime, PR_eventKernelStart, PR_eventKernelStop); \
	PR_kernelTime += kernelElapsedTime; \
	kernelElapsedTime; \
}) \
//

// TODO: put an _mpause
#define PR_waitKernel(args) \
	while(PR_WAIT_START_COND) { \
  	/*asm("" ::: "memory")*/pthread_yield(); \
  } \
	cudaStreamSynchronize(PR_streams[PR_currentStream]); \
	while(PR_WAIT_FINISH_COND) { /* this should not be needed */ \
		/*asm("" ::: "memory")*/ pthread_yield(); \
	} \
	PR_AFTER_WAIT(args); \
//

// Important: Only one file in the project must include pr-stm-internal.cuh!
//            Put custom implementation by re-defining *_EXT MACROs there.
// Example:
//   #undef PR_ARGS_S_EXT
//   #define PR_ARGS_S_EXT typedef struct /* your custom implementation */
// Then do:
// #include "pr-stm-internal.cuh" // implementation using your MACROs

// wrapper functions
PR_HOST void PR_init(int nbStreams); // call this before anything else

PR_HOST void PR_noStatistics(pr_tx_args_s *args);
PR_HOST void PR_createStatistics(pr_tx_args_s *args);
PR_HOST void PR_resetStatistics(pr_tx_args_s *args);
PR_HOST void PR_prepareIO(pr_tx_args_s *args, pr_buffer_s inBuf, pr_buffer_s outBuf);
PR_HOST void PR_run(void(*callback)(pr_tx_args_dev_host_s), pr_tx_args_s *args);

PR_HOST void PR_init_noCallback(int nbStreams);
PR_HOST void PR_prepare_noCallback(pr_tx_args_s *args);
PR_HOST void PR_postrun_noCallback(pr_tx_args_s *args);

PR_HOST cudaStream_t PR_getCurrentStream();
// Call this after retriving output to change to the next stream.
// This also calls PR_reduceCommitAborts to obtain statistics.
// It does not synchronize the stream, thus call it twice to force
// the copy of current data, e.g., before exiting.
PR_HOST void PR_useNextStream(pr_tx_args_s *args);

// resets aborts and commits into PR_nb*SinceCheckpoint
PR_HOST void PR_checkpointAbortsCommits();

PR_HOST void PR_retrieveIO(pr_tx_args_s *args); // updates PR_nbAborts and PR_nbCommits
PR_HOST void PR_disposeIO(pr_tx_args_s *args);

PR_HOST void PR_teardown(); // after you are done

PR_DEVICE void PR_beforeKernel_EXT(PR_txCallDefArgs);
PR_DEVICE void PR_afterKernel_EXT (PR_txCallDefArgs);
PR_DEVICE void PR_beforeStart_EXT(PR_txCallDefArgs);
PR_DEVICE void PR_afterCommit_EXT(PR_txCallDefArgs);

#endif /** __cplusplus **/

#endif /* PR_STM_H_GUARD */
