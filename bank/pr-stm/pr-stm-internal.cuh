#ifndef PR_STM_I_H_GUARD
#define PR_STM_I_H_GUARD

#include "pr-stm.cuh"

#include <curand_kernel.h>
#include <cuda_profiler_api.h>
#include "helper_cuda.h"
#include "helper_timer.h"

PR_ARGS_S_EXT;
PR_DEV_BUFF_S_EXT;

// TODO: define this elsewhere
#define SFENCE asm volatile ("sfence" ::: "memory")
// #define LFENCE asm volatile ("lfence" ::: "memory")
// #define MFENCE asm volatile ("mfence" ::: "memory")

// Global vars
PR_globalVars;

// ----------------------------------------
// Utility defines
#define PR_ADD_TO_ARRAY(array, size, value) ({ \
	(array)[(size)++] = (value); \
})

#define PR_ADD_TO_LOG(log, value) ({ \
	((void**)log.buf)[log.size++] = ((void*)value); \
})

// assume find ran first
#define PR_RWSET_GET_VAL(rwset, idx) \
	(rwset).values[idx] \
//

#define PR_RWSET_SET_VAL(rwset, idx, buf) \
	((rwset).values[idx] = buf) \
//

#define PR_RWSET_SET_VERSION(rwset, idx, version) \
	((rwset).versions[idx] = (int)version) \
//

#define PR_ADD_TO_RWSET(rwset, addr, version, value) \
	(rwset).addrs[(rwset).size] = (PR_GRANULE_T*)(addr); \
	PR_RWSET_SET_VAL(rwset, (rwset).size, value); \
	PR_RWSET_SET_VERSION(rwset, (rwset).size, version); \
	(rwset).size += 1; \
//

#define PR_FIND_IN_RWSET(rwset, addr) ({ \
	int i; \
	PR_GRANULE_T* a = (PR_GRANULE_T*)(addr); \
	long res = -1; \
	for (i = 0; i < (rwset).size; i++) { \
		if ((rwset).addrs[i] == a) { \
			res = i; \
			break; \
		} \
	} \
	res; \
})

// ----------------------------------------

//openRead Function reads data from global memory to local memory. args->rset_vals stores value and args->rset_versions stores version.
PR_DEVICE PR_GRANULE_T PR_i_openReadKernel(
	PR_args_s *args, PR_GRANULE_T *addr
) {
	int j, k;
	int temp, version;
	PR_GRANULE_T res;
	// volatile int *data = (volatile int*)args->data;
	// int target = args->rset_addrs[args->rset_size];

	temp = PR_GET_MTX(args->mtx, addr);
	// ------------------------------------------------------------------------
	// TODO: no repeated reads/writes
	k = PR_FIND_IN_RWSET(args->wset, addr);
	// ------------------------------------------------------------------------

	if (!args->is_abort && !PR_CHECK_PRELOCK(temp)) {
		// ------------------------------------------------------------------------
		// if (PR_THREAD_IDX == 405) printf("[405] did not abort yet!\n");
		// not locked
		if (k != -1) {
			// in wset
			res = PR_RWSET_GET_VAL(args->wset, k);
		} else {
			j = PR_FIND_IN_RWSET(args->rset, addr);
		// ------------------------------------------------------------------------
			version = PR_GET_VERSION(temp);
		// ------------------------------------------------------------------------
			if (j != -1) {
				// in rset
				// TODO: this seems a bit different in the paper
				res = PR_RWSET_GET_VAL(args->rset, j);
				PR_RWSET_SET_VERSION(args->rset, j, version);
			} else {
				// not found
		// ------------------------------------------------------------------------
				res = *addr;
				PR_ADD_TO_RWSET(args->rset, addr, version, res);
		// ------------------------------------------------------------------------
			}
		}
		// ------------------------------------------------------------------------
	} else {
		// if (PR_THREAD_IDX == 405) printf("[405] PR_CHECK_PRELOCK failed lock = 0x%x tid = %i version = %i addr = %p!\n",
		// 	temp, PR_GET_OWNER(temp), PR_GET_VERSION(temp), addr);
		res = *addr;
		args->is_abort = 1;
	}
	return res;
}

PR_DEVICE void PR_i_openWriteKernel(
	PR_args_s *args, PR_GRANULE_T *addr, PR_GRANULE_T wbuf
) {
	int j, k;
	int temp, version;

	temp = PR_GET_MTX(args->mtx, addr);
	if (!args->is_abort && !PR_CHECK_PRELOCK(temp)) {
		// ------------------------------------------------------------------------
		// // not locked --> safe to access TODO: the rset seems redundant
		// TODO: non-repeated writes
		j = PR_FIND_IN_RWSET(args->rset, addr); // check in the read-set
		k = PR_FIND_IN_RWSET(args->wset, addr); // check in the write-set
		// ------------------------------------------------------------------------
		version = PR_GET_VERSION(temp);

		// ------------------------------------------------------------------------
		// TODO: assume read-before-write
		if (j == -1) {
			// not in the read-set
			PR_ADD_TO_RWSET(args->rset, addr, version, wbuf);
		} else {
			// update the read-set (not the version) TODO: is this needed?
			PR_RWSET_SET_VAL(args->rset, j, wbuf);
		}
		// ------------------------------------------------------------------------

		// ------------------------------------------------------------------------
		if (k == -1) {
		// ------------------------------------------------------------------------
			// does not exist in write-set
			PR_ADD_TO_RWSET(args->wset, addr, version, wbuf);
		// ------------------------------------------------------------------------
		} else {
			// update the write-set
			PR_RWSET_SET_VAL(args->wset, j, wbuf);
		}
		// ------------------------------------------------------------------------
	} else {
		// already locked
		args->is_abort = 1;
	}
}

PR_DEVICE int PR_i_tryPrelock(PR_args_s *args)
{
	int i, res = 0;
	PR_GRANULE_T *addr;
	int version, lock;
	bool validVersion, notLocked, ownerHasHigherPriority;
	int tid = args->tid;

	for (i = 0; i < args->rset.size; i++) {
		addr = args->rset.addrs[i];
		version = args->rset.versions[i];
		lock = PR_GET_MTX(args->mtx, addr);
		validVersion = PR_GET_VERSION(lock) == version;
		notLocked = !PR_CHECK_PRELOCK(lock);
		ownerHasHigherPriority = PR_GET_OWNER(lock) < tid;
		if (validVersion && (notLocked || !ownerHasHigherPriority)) {
			res++;
			continue;
		} else {
			res = -1; // did not validate
			break;
		}
	}

	return res;
}

PR_DEVICE void PR_i_unlockWset(PR_args_s *args)
{
	int i;
	int *lock, lval, nval;
	int tid = args->tid;

	for (i = 0; i < args->wset.size; i++) {
		lock = (int*) &(PR_GET_MTX(args->mtx, args->wset.addrs[i]));
		lval = *lock;
		nval = lval & PR_MASK_VERSION;
		if (PR_GET_OWNER(lval) == tid) atomicCAS(lock, lval, nval);
	}
}

PR_DEVICE void
PR_i_validateKernel(PR_args_s *args)
{
	int vr = 0; // flag for how many values in read  set is still validate
	int vw = 0; // flag for how many values in write set is still validate
	int i, k;
	int *lock, *old_lock, new_lock, lval, oval, nval;
	int tid = args->tid;

	int isLocked, isPreLocked, ownerIsSelf, ownerHasHigherPriority, newVersion;

	if (args->is_abort) return;

	vr = PR_i_tryPrelock(args);
	if (vr == -1) {
		args->is_abort = 1;
		return; // abort
	}

	// __threadfence(); // The paper does not mention this fence

	for (i = 0; i < args->wset.size; i++) {
		while (1) {
			// spin until this thread can lock one account in write set

			lock = (int*) &(PR_GET_MTX(args->mtx, args->wset.addrs[i]));
			lval = *lock;

			//check if version changed or locked by higher priority thread
			isLocked = PR_CHECK_LOCK(lval);
			isPreLocked = PR_CHECK_PRELOCK(lval);
			ownerHasHigherPriority = PR_GET_OWNER(lval) < tid;
			newVersion = PR_GET_VERSION(lval) != args->wset.versions[i];

			if (isLocked || (isPreLocked && ownerHasHigherPriority) || newVersion) {
				// if one of accounts is locked by a higher priority thread
				// or version changed, unlock all accounts it already locked
				for (k = 0; k < i; k++) {
					old_lock = (int*) &(PR_GET_MTX(args->mtx, args->wset.addrs[k]));
					oval = *old_lock;

					// check if this account is still locked by itself
					isPreLocked = PR_CHECK_PRELOCK(oval);
					ownerIsSelf = PR_GET_OWNER(oval) == tid;
					if (isPreLocked && ownerIsSelf) {
						nval = oval & PR_MASK_VERSION;
						atomicCAS(old_lock, oval, nval);
					}
				}
				// if one of accounts is locked by a higher priority thread or version changed, return false
				args->is_abort = 1;
				return;
			}
			new_lock = PR_PRELOCK_VAL(args->wset.versions[i], tid);

			// atomic lock that account
			if (atomicCAS(lock, lval, new_lock) == lval) break;
		}
	}

	// __threadfence(); // The paper does not mention this fence
	// if this thread can pre-lock all accounts it needs to, really lock them
	for (i = 0; i < args->wset.size; i++) {
		// get lock|owner|version from global memory
	 	lock = (int*) &(PR_GET_MTX(args->mtx, args->wset.addrs[i]));
		lval = *lock;

		// if it is still locked by itself (check lock flag and owner position)
		ownerIsSelf = PR_GET_OWNER(lval) == tid;
		if (ownerIsSelf) {
			new_lock = PR_LOCK_VAL(args->wset.versions[i], tid); // temp final lock
			if (atomicCAS(lock, lval, new_lock) == lval) {
				vw++;	// if succeed, vw++
			} else {
				// failed to lock
				PR_i_unlockWset(args);
				args->is_abort = 1;
				return;
			}
		} else {
			// cannot final lock --> unlock all
			PR_i_unlockWset(args);
			args->is_abort = 1;
			return;
		}
	}
	if (vw == args->wset.size && vr == args->rset.size) {
		// all locks taken (should not need the extra if)
		PR_AFTER_VAL_LOCKS_EXT(args);
	}
}

PR_DEVICE void PR_i_commitKernel(PR_args_s *args)
{
	int i;
	int *lock, nval; // lval
	PR_GRANULE_T *addr, val;

	// write all values in write set back to global memory and increase version
	for (i = 0; i < args->wset.size; i++) {
		addr = args->wset.addrs[i];
		val = args->wset.values[i]; // TODO: variable size
		PR_SET(addr, val);
		PR_AFTER_WRITEBACK_EXT(args, i, addr, val);
	}
	__threadfence();
	for (i = 0; i < args->wset.size; i++) {
		// lval = PR_LOCK_VAL(args->wset.versions[i], PR_THREAD_IDX);
		lock = (int*) &(PR_GET_MTX(args->mtx, args->wset.addrs[i]));
		// increase version(if version is going to overflow, change it to 0)
		nval = args->wset.versions[i] < PR_LOCK_VERSION_OVERFLOW ?
			(args->wset.versions[i] + 1) << PR_LOCK_VERSION_BITS :
			0;
		// atomicCAS(lock, lval, nval);
		*lock = nval;
	}
}

// Be sure to match the kernel config!
PR_ENTRY_POINT void PR_reduceCommitAborts(
	int doReset,
	int targetStream,
	PR_globalKernelArgs,
	uint64_t *nbCommits,
	uint64_t *nbAborts)
{
	const int      WARP_SIZE         = 32;
	const uint32_t FULL_MASK         = 0xffffffff;
	const int      MAX_SHARED_MEMORY = 32;
	__shared__ uint64_t sharedSumCommits[MAX_SHARED_MEMORY];
	__shared__ uint64_t sharedSumAborts [MAX_SHARED_MEMORY];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int idx = tid + blockDim.x*gridDim.x*targetStream;
	uint64_t tmpAborts  = args.nbAborts [idx];
	uint64_t tmpCommits = args.nbCommits[idx];
	uint32_t mask;
	int writerTid;
	int writerPos;

	if (doReset) {
		args.nbAborts [idx] = 0;
		args.nbCommits[idx] = 0;
	}

	mask = __ballot_sync(FULL_MASK, 1);

	#pragma unroll (4)
	for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
		tmpAborts  += __shfl_xor_sync(mask, tmpAborts,  offset);
		tmpCommits += __shfl_xor_sync(mask, tmpCommits, offset);
	}

	writerTid = threadIdx.x % WARP_SIZE;
	writerPos = threadIdx.x / WARP_SIZE;

	if (writerTid == 0) {
		sharedSumAborts [writerPos] = tmpAborts ;
		sharedSumCommits[writerPos] = tmpCommits;
	}
	// Shared memory must be synchronized via barriers
	__syncthreads();
	if (writerTid == 0) {
		mask = __ballot_sync(FULL_MASK, 1); // sync only threads that reach here
		if (threadIdx.x == 0) {
			// first thread in the entire block sums the rest
			tmpAborts  = sharedSumAborts [0];
			tmpCommits = sharedSumCommits[0];
			int divWarpSize = blockDim.x / WARP_SIZE;
			int modWarpSize = blockDim.x % WARP_SIZE > 0 ? 1 : 0;
			divWarpSize += modWarpSize;
			#pragma unroll (16)
			for (int i = 1; i < divWarpSize; i++) {
				tmpAborts  += sharedSumAborts [i];
				tmpCommits += sharedSumCommits[i];
			}
			// Global memory must be synchronized via atomics
			atomicAdd((unsigned long long*)nbAborts,  (unsigned long long)tmpAborts);
			atomicAdd((unsigned long long*)nbCommits, (unsigned long long)tmpCommits);
		}
	}
}

PR_HOST void
PR_i_afterStream(cudaStream_t stream, cudaError_t status, void *data)
{
	PR_isDone[PR_currentStream] = 1;
	PR_isStart[PR_currentStream] = 0;
	SFENCE;
}


PR_HOST void PR_noStatistics(pr_tx_args_s *args)
{
	args->host.nbAborts = NULL;
	args->host.nbCommits = NULL;
	args->dev.nbAborts = NULL;
	args->dev.nbCommits = NULL;
}

PR_HOST void PR_createStatistics(pr_tx_args_s *args)
{
	int nbThreads = PR_blockNum * PR_threadNum;
	int sizeArray = nbThreads * sizeof(int);

	args->host.nbAborts = NULL; // not used
	args->host.nbCommits = NULL;

	PR_ALLOC(args->dev.nbAborts, sizeArray*PR_streamCount);
	PR_ALLOC(args->dev.nbCommits, sizeArray*PR_streamCount);

	PR_resetStatistics(args);
}

PR_HOST void PR_resetStatistics(pr_tx_args_s *args)
{
	int nbThreads = PR_blockNum * PR_threadNum;
	int sizeArray = nbThreads * sizeof(int);
	cudaStream_t stream = PR_streams[PR_currentStream];

	PR_CHECK_CUDA_ERROR(cudaMemsetAsync(args->dev.nbAborts, 0, sizeArray*PR_streamCount, stream), "");
	PR_CHECK_CUDA_ERROR(cudaMemsetAsync(args->dev.nbCommits, 0, sizeArray*PR_streamCount, stream), "");
	PR_nbCommits = 0;
	PR_nbAborts = 0;
}

PR_HOST void PR_prepareIO(
	pr_tx_args_s *args,
	pr_buffer_s inBuf,
	pr_buffer_s outBuf
) {
	// input
	args->dev.inBuf = inBuf.buf;
	args->dev.inBuf_size = inBuf.size;
	args->host.inBuf = NULL; // Not needed
	args->host.inBuf_size = 0;

	// output
	args->dev.outBuf = outBuf.buf;
	args->dev.outBuf_size = outBuf.size;
	args->host.outBuf = NULL; // Not needed
	args->host.outBuf_size = 0;
}

PR_HOST void PR_i_cudaPrepare(
	pr_tx_args_s *args,
	void(*callback)(pr_tx_args_dev_host_s)
) {
	args->host.mtx = PR_lockTableHost;
	args->host.current_stream = PR_currentStream;

	// dev is a CPU-local struct that is passed to the kernel
	args->dev.mtx  = PR_lockTableDev;
	args->dev.current_stream = PR_currentStream;
	args->callback = callback;
}

PR_HOST void PR_prepare_noCallback(
	pr_tx_args_s *args
) {
	pr_buffer_s a,b;
	args->host.mtx = PR_lockTableHost;
	args->host.current_stream = PR_currentStream;

	// dev is a CPU-local struct that is passed to the kernel
	args->dev.mtx  = PR_lockTableDev;
	args->dev.current_stream = PR_currentStream;
	PR_createStatistics(args);
	PR_prepareIO(args, a, b);
	// StopWatchInterface *kernelTime = NULL;
	cudaStream_t stream = PR_streams[PR_currentStream];

	// cudaFuncSetCacheConfig(args->callback, cudaFuncCachePreferL1);

	PR_isDone[PR_currentStream] = 0;
	PR_isStart[PR_currentStream] = 1;
	SFENCE;
	// pass struct by value
	cudaEventRecord(PR_eventKernelStart, stream); // TODO: does not work in multi-thread
}
PR_HOST void PR_postrun_noCallback(
	pr_tx_args_s *args
) {
	cudaStream_t stream = PR_streams[PR_currentStream];
	cudaEventRecord(PR_eventKernelStop, stream);
	CUDA_CHECK_ERROR(cudaStreamAddCallback(
		stream, PR_i_afterStream, NULL, 0
	), "");
};

PR_HOST void PR_retrieveIO(pr_tx_args_s *args)
{
	int i, nbThreads = PR_blockNum * PR_threadNum;
	int sizeArray = nbThreads * sizeof(int);
	// int sizeMtx = PR_LOCK_TABLE_SIZE * sizeof(int);
	static int *hostNbAborts;
	static int *hostNbCommits;
	static int last_size;
	cudaStream_t stream = PR_streams[PR_currentStream];

	if (hostNbAborts == NULL) {
		cudaMallocHost(&hostNbAborts, sizeArray*PR_streamCount);
		cudaMallocHost(&hostNbCommits, sizeArray*PR_streamCount);
		last_size = sizeArray;
	}

	if (last_size != sizeArray) {
		// TODO
		hostNbAborts = (int*)realloc(hostNbAborts, sizeArray*PR_streamCount);
		hostNbCommits = (int*)realloc(hostNbCommits, sizeArray*PR_streamCount);
		last_size = sizeArray;
	}

	// TODO: this is only done in the end --> need to copy all streams!

	void *devNbAborts = (int*)args->dev.nbAborts;
	void *devNbCommits = (int*)args->dev.nbCommits;

	// PR_CPY_TO_HOST(PR_lockTableHost, PR_lockTableDev, sizeMtx); // TODO: only for debug
	PR_CPY_TO_HOST_ASYNC(hostNbAborts, devNbAborts, sizeArray*PR_streamCount, stream);
	PR_CPY_TO_HOST_ASYNC(hostNbCommits, devNbCommits, sizeArray*PR_streamCount, stream);
	PR_nbAborts = 0;
	PR_nbCommits = 0;
	CUDA_CHECK_ERROR(cudaStreamSynchronize(PR_getCurrentStream()), "");
	for (i = 0; i < nbThreads*PR_streamCount; ++i) {
		PR_nbAborts += hostNbAborts[i];
		PR_nbCommits += hostNbCommits[i];
	}
	// printf("PR_nbAborts = %lli\n", PR_nbAborts);
}

PR_HOST cudaStream_t PR_getCurrentStream()
{
	return PR_streams[PR_currentStream];
}

PR_HOST void PR_disposeIO(pr_tx_args_s *args)
{
	PR_CHECK_CUDA_ERROR(cudaFree(args->dev.nbAborts), "");
	PR_CHECK_CUDA_ERROR(cudaFree(args->dev.nbCommits), "");
}
PR_HOST void
PR_i_afterStats(cudaStream_t stream, cudaError_t status, void *data)
{
	uintptr_t curStream = (uintptr_t)data;

	PR_nbCommitsLastKernel = *PR_sumNbCommits - PR_nbCommitsStrm[curStream];
	PR_nbAbortsLastKernel  = *PR_sumNbAborts  - PR_nbAbortsStrm[curStream];

	PR_nbCommitsSinceCheckpoint += PR_nbCommitsLastKernel;
	PR_nbAbortsSinceCheckpoint  += PR_nbAbortsLastKernel;

	PR_nbCommitsStrm[curStream] = *PR_sumNbCommits;
	PR_nbAbortsStrm[curStream]  = *PR_sumNbAborts;
	SFENCE;
}

PR_HOST void
PR_i_run(pr_tx_args_s *args)
{
	// StopWatchInterface *kernelTime = NULL;
	cudaStream_t stream = PR_streams[PR_currentStream];

	// cudaFuncSetCacheConfig(args->callback, cudaFuncCachePreferL1);

	PR_isDone[PR_currentStream] = 0;
	PR_isStart[PR_currentStream] = 1;
	SFENCE;
	// pass struct by value
	cudaEventRecord(PR_eventKernelStart, stream); // TODO: does not work in multi-thread
	// if (stream != NULL) {
		args->callback<<< PR_blockNum, PR_threadNum, 0, stream >>>(args->dev);
	// } else {
	// 	// serializes
	// 	args->callback<<< PR_blockNum, PR_threadNum >>>(args->dev);
	// }
	cudaEventRecord(PR_eventKernelStop, stream);
	CUDA_CHECK_ERROR(cudaStreamAddCallback(
		stream, PR_i_afterStream, NULL, 0
	), "");
}

// Extension wrapper
PR_HOST void PR_init(int nbStreams)
{
	const size_t sizeMtx = PR_LOCK_TABLE_SIZE * sizeof(int);

	PR_streamCount = nbStreams;
	if (PR_streams == NULL) {
		PR_streams = (cudaStream_t*)malloc(sizeof(cudaStream_t) * nbStreams);
		for (int i = 0; i < nbStreams; ++i) {
			PR_CHECK_CUDA_ERROR(cudaStreamCreate(PR_streams + i), "stream");
		}
	}

	if (PR_lockTableDev == NULL) {
		cudaEventCreate(&PR_eventKernelStart);
		cudaEventCreate(&PR_eventKernelStop);
		PR_lockTableHost = NULL; // TODO: host locks not needed
		// memset(PR_lockTableHost, 0, PR_LOCK_TABLE_SIZE * sizeof(int));
		PR_ALLOC(PR_lockTableDev, sizeMtx);
		PR_CHECK_CUDA_ERROR(cudaMemset(PR_lockTableDev, 0, sizeMtx), "memset");
	}

	if (PR_sumNbCommitsDev == NULL) {
		size_t sizeCount = sizeof(uint64_t)*2;
		CUDA_DEV_ALLOC (PR_sumNbCommitsDev, sizeCount);
		CUDA_HOST_ALLOC(PR_sumNbCommits,    sizeCount);
		PR_sumNbAbortsDev = PR_sumNbCommitsDev + 1;
		PR_sumNbAborts    = PR_sumNbCommits    + 1;

		PR_CHECK_CUDA_ERROR(cudaMemset(PR_sumNbCommitsDev, 0, sizeCount), "");

		*PR_sumNbCommits = 0;
		*PR_sumNbAborts  = 0;
	}
}
PR_HOST void PR_init_noCallback(int nbStreams)
{
	PR_init(nbStreams);
}
PR_HOST void PR_run(void(*callback)(pr_tx_args_dev_host_s), pr_tx_args_s *pr_args)
{
	PR_i_cudaPrepare(pr_args, callback);
	PR_BEFORE_RUN_EXT(pr_args);
	PR_i_run(pr_args);
	PR_AFTER_RUN_EXT(pr_args);
}
PR_HOST void PR_teardown()
{
	// cudaFreeHost((void*)PR_lockTableHost);
	cudaFree((void*)PR_lockTableDev);
	for (int i = 0; i < PR_streamCount; ++i) {
		PR_CHECK_CUDA_ERROR(cudaStreamDestroy(PR_streams[i]), "stream");
	}
	free(PR_streams);
}
PR_HOST void PR_useNextStream(pr_tx_args_s *args)
{
	static size_t countSize = sizeof(uint64_t)*2;
	uintptr_t curStream = PR_currentStream;

	CUDA_CHECK_ERROR(cudaGetLastError(), "before PR_reduceCommitAborts");

	PR_reduceCommitAborts<<<PR_blockNum, PR_threadNum, 0, PR_streams[PR_currentStream]>>>
		(0, PR_currentStream, args->dev, (uint64_t*)PR_sumNbCommitsDev, (uint64_t*)PR_sumNbAbortsDev);

	CUDA_CHECK_ERROR(cudaGetLastError(), "PR_reduceCommitAborts");

	// not first time, copy previous data
	CUDA_CPY_TO_HOST_ASYNC(
		PR_sumNbCommits,
		PR_sumNbCommitsDev,
		countSize,
		PR_streams[PR_currentStream]
	);
	CUDA_CHECK_ERROR(cudaStreamAddCallback(
		PR_streams[PR_currentStream], PR_i_afterStats, (void*)curStream, 0
	), "");

	PR_CHECK_CUDA_ERROR(cudaMemsetAsync(PR_sumNbCommitsDev, 0, countSize, PR_streams[PR_currentStream]), "");

	PR_currentStream = (PR_currentStream + 1) % PR_streamCount;
}
PR_HOST void PR_checkpointAbortsCommits()
{
	PR_nbAbortsSinceCheckpoint = 0;
	PR_nbCommitsSinceCheckpoint = 0;
}

PR_DEVICE void PR_beforeKernel_EXT(PR_txCallDefArgs) { PR_BEFORE_KERNEL_EXT(args, pr_args); }
PR_DEVICE void PR_afterKernel_EXT (PR_txCallDefArgs) { PR_AFTER_KERNEL_EXT(args, pr_args);  }
PR_DEVICE void PR_beforeBegin_EXT (PR_txCallDefArgs) { PR_BEFORE_BEGIN_EXT(args, pr_args);  }
PR_DEVICE void PR_afterCommit_EXT (PR_txCallDefArgs) { PR_AFTER_COMMIT_EXT(args, pr_args);  }

#endif /* PR_STM_I_H_GUARD */
