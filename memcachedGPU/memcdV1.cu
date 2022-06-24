// TODO: includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define PR_MAX_RWSET_SIZE 256

#include "pr-stm.cuh"
#include "pr-stm-internal.cuh"
#include "keyGenAndCacheStore.h"

#define VAL_SIZE   8
#define KEY_SIZE   1

#define USE_ZIPF      1
#define ZIPF_PARAM  0.5 // TODO: this is not standard Zipf parameter

#define RAND_R_FNC(seed) ({ \
    uint64_t next = seed; \
    uint64_t result; \
    next *= 1103515245; \
    next += 12345; \
    result = (uint64_t) (next / 65536) % 2048; \
    next *= 1103515245; \
    next += 12345; \
    result <<= 10; \
    result ^= (uint64_t) (next / 65536) % 1024; \
    next *= 1103515245; \
    next += 12345; \
    result <<= 10; \
    result ^= (uint64_t) (next / 65536) % 1024; \
    seed = next; \
    result; \
})

// ===========================
// Structure of the cache:
// key:   [ ... key_part_1 ... | ... key_part_2 ... | ... ]
// value: [ ... val_part_1 ... | ... val_part_2 ... | ... ]
// ts:    [ ... entry_ts ... ]
// state: [ ... entry is INVALID/VALID/READ/WRITTEN ... ]
// key has KEY_SIZE parts and value has VAL_SIZE parts
// ===========================

typedef int granule_t;

typedef struct get_output_ {
	int isFound;
	granule_t val[VAL_SIZE];
} get_output_t;

typedef struct memcd_ {
	granule_t *key;           /* keys in global memory */
	granule_t *val;           /* values in global memory */
	granule_t *ts;            /* last access TS in global memory */
	granule_t *state;         /* state in global memory */
  long seed;
  long nbKeys;
  int nbSets;
  int nbWays;
  int nbBlocks;
  int nbThreads;
  int nbReps;
	float probReadKernel;
	int nbTXsPerThread;
  int *curr_clock;
  get_output_t *output;     /* only for the GET kernel */
  granule_t *input_keys;    /* target input keys, one per thread */
  granule_t *input_vals;    /* only for the SET kernel, one per thread */
} memcd_s;

enum { // state of a memcd cache entry
  INVALID     = 0,
  VALID       = 1,
  READ        = 2,
  WRITTEN     = 4
};

// ===========================
// global info of benchmark
memcd_s memcd_host, memcd_dev;
// ===========================

__global__ void memcdReadTx(PR_globalKernelArgs /* args */, memcd_s input)
{
	int tid = PR_THREAD_IDX;
	PR_enterKernel(tid);

	int nbSets = input.nbSets;
	int nbWays = input.nbWays;
	int nbTXsPerThread = input.nbTXsPerThread;
	int id = tid;
	int wayId = id % (nbWays /*+ devParsedData.trans*/);
	int targetKey = id / (nbWays + nbTXsPerThread); // id of the key that each thread will take

	// num_ways threads will colaborate for the same input
	// REQUIREMENT: 1 block >= num_ways

	granule_t       *keys = input.key;
	granule_t     *values = input.val;
	granule_t *timestamps = input.ts;
	granule_t      *state = input.state;

	// TODO: out is NULL
	get_output_t        *out = input.output;
	int           curr_clock = *((int*)input.curr_clock);
	granule_t    *input_keys = input.input_keys;

	// __shared__ int aborted[1024];
	// if (wayId == 0) aborted[targetKey] = 0;
	// __syncthreads();

	for (int i = 0; i < nbTXsPerThread; ++i) { // num_ways keys
		out[id*nbTXsPerThread + i].isFound = 0;
	}

	for (int i = 0; i < (nbWays + nbTXsPerThread); ++i) { // num_ways keys
		// TODO: for some reason input_key == 0 does not work --> PR-STM loops forever
		int input_key = input_keys[targetKey + i]; // input size is well defined
		int target_set = input_key % nbSets;
		int thread_pos = target_set*nbWays + wayId;
		int thread_is_found;
		granule_t thread_key;
		granule_t thread_val;
		granule_t thread_state;

		// PR_write(&timestamps[thread_pos], curr_clock);
		thread_key = keys[thread_pos];
		thread_state = state[thread_pos];

		__syncthreads(); // TODO: each num_ways thread helps on processing the targetKey

		// TODO: divergency here
		thread_is_found = (thread_key == input_key && ((thread_state & VALID) != 0));
		if (thread_is_found) {
			int nbRetries = 0;
			int ts;
			PR_txBegin();
			if (nbRetries > 1024) {
				// i--; // TODO: should test the key again
				// aborted[targetKey] = 1;
				// printf("Thread%i blocked thread_key=%i thread_pos=%i ts=%i curr_clock=%i\n", id, thread_key, thread_pos, ts, curr_clock);
				break; // retry, key changed
			}
			nbRetries++;

			PR_read(&keys[thread_pos]);
			// // TODO:
			// if (thread_key != PR_read(&keys[thread_pos])) {
			// 	i--; // retry, key changed
			// 	break;
			// }
			thread_val = PR_read(&values[thread_pos]);
			ts = PR_read(&timestamps[thread_pos]); // assume read-before-write
			if (ts < curr_clock && nbRetries < 5) PR_write(&timestamps[thread_pos], curr_clock);
			else timestamps[thread_pos] = curr_clock; // TODO: cannot transactionally write this...

			PR_txCommit();

			out[targetKey + i].isFound = 1;
			out[targetKey + i].val[0] = thread_val; // TODO: value_size
		}
		// if (aborted[targetKey]) {
		// 	// i--; // repeat this loop // TODO: blocks forever
		// }
		// aborted[targetKey] = 0;
		// __syncthreads();
	}

	PR_exitKernel();
}

// TODO: IMPORTANT ---> set PR_MAX_RWSET_SIZE to number of ways

__global__ void memcdWriteTx(PR_globalKernelArgs /* args */, memcd_s input)
{
	int tid = PR_THREAD_IDX;
	PR_enterKernel(tid);

	// TODO: blockDim.x must be multiple of num_ways --> else this does not work
	int nbSets = input.nbSets;
	int nbWays = input.nbWays;
	int nbTXsPerThread = input.nbTXsPerThread;
	int id = tid; //threadIdx.x+blockDim.x*blockIdx.x;

	// TODO: too much memory (this should be blockDim.x / num_ways)
	// TODO: 32 --> min num_ways == 8 for 256 block size
	// TODO: I'm using warps --> max num_ways is 32 (CAN BE EXTENDED!)
	const int maxWarpSlices = 32; // 32*32 == 1024
	int warpSliceID = threadIdx.x / nbWays;
	int wayId = id % (nbWays /*+ devParsedData.trans*/);
	int reductionID = wayId / 32;
	int reductionSize = max(nbWays / 32, 1);
	int targetKey = id / (nbWays + nbTXsPerThread); // id of the key that each group of num_ways thread will take

	__shared__ int reduction_is_found[maxWarpSlices]; // TODO: use shuffle instead
	__shared__ int reduction_is_empty[maxWarpSlices]; // TODO: use shuffle instead
	__shared__ int reduction_empty_min_id[maxWarpSlices]; // TODO: use shuffle instead
	__shared__ int reduction_min_ts[maxWarpSlices]; // TODO: use shuffle instead

	// num_ways threads will colaborate for the same input
	// REQUIREMENT: 1 block >= num_ways

	__shared__ int failed_to_insert[256]; // TODO
	if (wayId == 0) failed_to_insert[warpSliceID] = 0;

	get_output_t     *out = (get_output_t*)input.output;
	granule_t       *keys = input.key;
	granule_t     *values = input.val;
	granule_t *timestamps = input.ts;
	granule_t      *state = input.state;
	int           curr_clock = *((int*)input.curr_clock);
	int          *input_keys = input.input_keys;
	int          *input_vals = input.input_vals;

	int thread_is_found; // TODO: use shuffle instead
	int thread_is_empty; // TODO: use shuffle instead
	// int thread_is_older; // TODO: use shuffle instead
	granule_t thread_key;
	// granule_t thread_val;
	granule_t thread_ts;
	granule_t thread_state;

	int checkKey;
	int maxRetries = 0;

	for (int i = 0; i < nbWays + nbTXsPerThread; ++i) {

		__syncthreads(); // TODO: check with and without this
		// TODO
		if (failed_to_insert[warpSliceID] && maxRetries < 64) { // TODO: blocks
			maxRetries++;
			i--;
		}
		__syncthreads(); // TODO: check with and without this
		if (wayId == 0) failed_to_insert[warpSliceID] = 0;

		// TODO: problem with the GET
		int input_key = input_keys[targetKey + i]; // input size is well defined
		int input_val = input_vals[targetKey + i]; // input size is well defined
		int target_set = input_key % nbSets;
		int thread_pos = target_set*nbWays + wayId;

		thread_key = keys[thread_pos];
		// thread_val = values[thread_pos]; // assume read-before-write
		thread_state = state[thread_pos];
		thread_ts = timestamps[thread_pos]; // TODO: only needed for evict

		// TODO: divergency here
		thread_is_found = (thread_key == input_key && (thread_state & VALID));
		thread_is_empty = !(thread_state & VALID);
		int empty_min_id = thread_is_empty ? id : id + 32; // warpSize == 32
		int min_ts = thread_ts;

		int warp_is_found = thread_is_found; // 1 someone has found; 0 no one found
		int warp_is_empty = thread_is_empty; // 1 someone has empty; 0 no empties
		const int FULL_MASK = -1;
		int mask = nbWays > 32 ? FULL_MASK : ((1 << nbWays) - 1) << (warpSliceID*nbWays);

		for (int offset = max(nbWays, 32)/2; offset > 0; offset /= 2) {
			warp_is_found = max(warp_is_found, __shfl_xor_sync(mask, warp_is_found, offset));
			warp_is_empty = max(warp_is_empty, __shfl_xor_sync(mask, warp_is_empty, offset));
			empty_min_id = min(empty_min_id, __shfl_xor_sync(mask, empty_min_id, offset));
			min_ts = min(min_ts, __shfl_xor_sync(mask, min_ts, offset));
		}

		reduction_is_found[reductionID] = warp_is_found;
		reduction_is_empty[reductionID] = warp_is_empty;
		reduction_empty_min_id[reductionID] = empty_min_id;
		reduction_min_ts[reductionID] = min_ts;

		// STEP: for n-way > 32 go to shared memory and try again
		warp_is_found = reduction_is_found[wayId % reductionSize];
		warp_is_empty = reduction_is_empty[wayId % reductionSize];
		empty_min_id = reduction_empty_min_id[wayId % reductionSize];
		min_ts = reduction_min_ts[wayId % reductionSize];

		for (int offset = reductionSize/2; offset > 0; offset /= 2) {
			warp_is_found = max(warp_is_found, __shfl_xor_sync(mask, warp_is_found, offset));
			warp_is_empty = max(warp_is_empty, __shfl_xor_sync(mask, warp_is_empty, offset));
			empty_min_id = min(empty_min_id, __shfl_xor_sync(mask, empty_min_id, offset));
			min_ts = min(min_ts, __shfl_xor_sync(mask, min_ts, offset));
		}

				// if (maxRetries == 8191) {
				// 	 printf("thr%i retry 8191 times for key%i thread_pos=%i check_key=%i \n",
				// 		id, input_key, thread_pos, checkKey);
				// }

		if (thread_is_found) {
			int nbRetries = 0; //TODO: on fail should repeat the search
			PR_txBegin(); // TODO: I think this may not work
			if (nbRetries > 0) {
				// TODO: is ignoring the input
				// someone got it; need to find a new spot for the key
				failed_to_insert[warpSliceID] = 1;
				break;
			}
			nbRetries++;

			checkKey = PR_read(&keys[thread_pos]); // read-before-write
			// TODO: does not work
			// if (checkKey != input_key) { // we are late
			// 	failed_to_insert[warpSliceID] = 1;
			// 	break;
			// }
			PR_read(&values[thread_pos]); // read-before-write
			PR_read(&timestamps[thread_pos]); // read-before-write
			// TODO: check if values changed: if yes abort
			PR_write(&keys[thread_pos], input_key);
			PR_write(&values[thread_pos], input_val);
			PR_write(&timestamps[thread_pos], curr_clock);
			// TODO: it seems not to reach this if but the nbRetries is needed
						// if (nbRetries == 8191) printf("thr%i aborted 8191 times for key%i thread_pos=%i rsetSize=%lu, wsetSize=%lu\n",
						// 	id, input_key, thread_pos, pr_args.rset.size, pr_args.wset.size);
			PR_txCommit();
			out[targetKey + i].isFound = 1;
			out[targetKey + i].val[0] = checkKey; // TODO: val size
		}

		// if(id == 0) printf("is found=%i\n", thread_is_found);

		// TODO: if num_ways > 32 this does not work very well... (must use shared memory)
		//       using shared memory --> each warp compute the min then: min(ResW1, ResW2)
		//       ResW1 and ResW2 are shared
		// was it found?
		if (!warp_is_found && thread_is_empty && empty_min_id == id) {
			// the low id thread must be the one that writes
			int nbRetries = 0;  //TODO: on fail should repeat the search
			PR_txBegin(); // TODO: I think this may not work
			if (nbRetries > 0) {
				// someone got it; need to find a new spot for the key
				failed_to_insert[warpSliceID] = 1;
				break;
			}
			nbRetries++;

			checkKey = PR_read(&keys[thread_pos]); // read-before-write
			// TODO: does not work
			// if (checkKey != input_key) { // we are late
			// 	failed_to_insert[warpSliceID] = 1;
			// 	break;
			// }
			PR_read(&values[thread_pos]); // read-before-write
			PR_read(&timestamps[thread_pos]); // read-before-write
			PR_read(&state[thread_pos]); // read-before-write
			// TODO: check if values changed: if yes abort
			PR_write(&keys[thread_pos], input_key);
			PR_write(&values[thread_pos], input_val);
			PR_write(&timestamps[thread_pos], curr_clock);
			int newState = VALID|WRITTEN;
			PR_write(&state[thread_pos], newState);
			PR_txCommit();
			out[targetKey + i].isFound = 0;
			out[targetKey + i].val[0] = checkKey; // TODO: val size
		}

		// not found, none empty --> evict the oldest
		if (!warp_is_found && !warp_is_empty && min_ts == thread_ts) {
			int nbRetries = 0; //TODO: on fail should repeat the search
			PR_txBegin(); // TODO: I think this may not work
			if (nbRetries > 0) {
		 		// someone got it; need to find a new spot for the key
				failed_to_insert[warpSliceID] = 1;
				break;
			}
			nbRetries++;

			checkKey = PR_read(&keys[thread_pos]); // read-before-write
			// TODO: does not work
			// if (checkKey != input_key) { // we are late
			// 	failed_to_insert[warpSliceID] = 1;
			// 	break;
			// }
			PR_read(&values[thread_pos]); // read-before-write
			PR_read(&timestamps[thread_pos]); // read-before-write
			// TODO: check if values changed: if yes abort
			PR_write(&keys[thread_pos], input_key);
			PR_write(&values[thread_pos], input_val);
			PR_write(&timestamps[thread_pos], curr_clock);
			// if (nbRetries == 8191) printf("thr%i aborted 8191 times for key%i thread_pos=%i rsetSize=%lu, wsetSize=%lu\n",
			// 	id, input_key, thread_pos, pr_args.rset.size, pr_args.wset.size);
			PR_txCommit();
			out[targetKey + i].isFound = 0;
			out[targetKey + i].val[0] = checkKey; // TODO: val size
		}
	}

	PR_exitKernel();
}
//

static void fillInputs(int idx, int total, long long item) {
	for (int j = 0; j < KEY_SIZE; ++j) {
		memcd_host.input_keys[idx + j*total] = (int)item;
	}
	for (int j = 0; j < VAL_SIZE; ++j) {
		memcd_host.input_vals[idx + j*total] = (int)item;
	}
}

int main(int argc, char **argv)
{
	// ===========================
	// arguments setup
	const char APP_HELP[] = ""                
	  "argument order:                     \n"
	  "  1) rng seed                       \n"
	  "  2) nb sets in cache               \n"
	  "  3) nb ways per set                \n"
	  "  4) key space (nb of possible keys)\n"
	  "  5) kernel config - nb blocks      \n"
	  "  6) kernel config - nb threads     \n"
	  "  7) nb TXs per thread              \n"
	  "  8) prob read TX kernel            \n"
		"  9) nb repetitions                 \n"
	"";
	const int NB_ARGS = 10;
	int argCnt = 1;
	
	if (argc != NB_ARGS) {
		printf("%s\n", APP_HELP);
		exit(EXIT_SUCCESS);
	}

	long    seed = memcd_host.seed           = (long)atol(argv[argCnt++]);
	int   nbSets = memcd_host.nbSets         = (int)atol(argv[argCnt++]);
	int   nbWays = memcd_host.nbWays         = (int)atol(argv[argCnt++]);
	long  nbKeys = memcd_host.nbKeys         = (long)atol(argv[argCnt++]);
	int   nbBlks = memcd_host.nbBlocks       = (int)atol(argv[argCnt++]);
	int   nbThrs = memcd_host.nbThreads      = (int)atol(argv[argCnt++]);
	int   nbTXsT = memcd_host.nbTXsPerThread = (int)atol(argv[argCnt++]);
	float prRead = memcd_host.probReadKernel = (float)atof(argv[argCnt++]);
	int   nbReps = memcd_host.nbReps         = (int)atol(argv[argCnt++]);

	struct timespec tTot1, tTot2, t1, t2;
	double tKernel_ms = 0.0, totT_ms = 0.0;
	int nbIters = 0;
	// ===========================

	// select GPU device
	CUDA_CHECK_ERROR(cudaSetDevice(0), "GPU select");

	// ===========================
	// memory setup
	memcpy(&memcd_dev, &memcd_host, sizeof(memcd_dev));

	CUDA_DUAL_ALLOC(memcd_host.key, memcd_dev.key, sizeof(granule_t)*nbSets*nbWays*KEY_SIZE);
	CUDA_DUAL_ALLOC(memcd_host.val, memcd_dev.val, sizeof(granule_t)*nbSets*nbWays*VAL_SIZE);
	CUDA_DUAL_ALLOC(memcd_host.state, memcd_dev.state, sizeof(granule_t)*nbSets*nbWays);
	CUDA_DUAL_ALLOC(memcd_host.ts, memcd_dev.ts, sizeof(granule_t)*nbSets*nbWays);
	CUDA_DUAL_ALLOC(memcd_host.curr_clock, memcd_dev.curr_clock, sizeof(int));
	CUDA_DUAL_ALLOC(memcd_host.input_keys, memcd_dev.input_keys, sizeof(granule_t)*nbThrs*nbBlks*nbTXsT*KEY_SIZE);
	CUDA_DUAL_ALLOC(memcd_host.input_vals, memcd_dev.input_vals, sizeof(granule_t)*nbThrs*nbBlks*nbTXsT*VAL_SIZE);
	CUDA_DUAL_ALLOC(memcd_host.output, memcd_dev.output, sizeof(get_output_t)*nbThrs*nbBlks*nbTXsT);

	*memcd_host.curr_clock = 1;
	memset(memcd_host.state, 0, sizeof(granule_t)*nbSets*nbWays);

	// ===========================

	// ===========================
	// Main loop
	PR_init(1);
	pr_tx_args_s args;

#ifdef USE_ZIPF
	fill_array_with_items(nbTXsT*nbBlks*nbThrs, nbKeys, ZIPF_PARAM, /* TODO: implement me */fillInputs);
#else
	// gen keys
	for (int i = 0; i < nbTXsT*nbBlks*nbThrs; ++i) {
		granule_t key = RAND_R_FNC(seed) % nbKeys;
		for (int j = 0; j < KEY_SIZE; ++j) {
			memcd_host.input_keys[i + j*nbTXsT*nbBlks*nbThrs] = key;
		}
		for (int j = 0; j < VAL_SIZE; ++j) {
			memcd_host.input_vals[i + j*nbTXsT*nbBlks*nbThrs] = key;
		}
	}
#endif

	// populates memory structs
	if (loadArray("cache_state", nbSets*nbWays, sizeof(granule_t), memcd_host.state)) {
		// file not found --> populate
		CUDA_CPY_TO_DEV(memcd_dev.state, memcd_host.state, sizeof(granule_t)*nbSets*nbWays);
		CUDA_CPY_TO_DEV(memcd_dev.curr_clock, memcd_host.curr_clock, sizeof(int));
		CUDA_CPY_TO_DEV(memcd_dev.input_keys, memcd_host.input_keys, sizeof(granule_t)*nbTXsT*nbBlks*nbThrs*KEY_SIZE);
		CUDA_CPY_TO_DEV(memcd_dev.input_vals, memcd_host.input_vals, sizeof(granule_t)*nbTXsT*nbBlks*nbThrs*VAL_SIZE);

		// Populates the cache
		PR_prepare_noCallback(&args);
		memcdWriteTx<<<nbBlks, nbThrs>>>(args.dev, memcd_dev);
		PR_postrun_noCallback(&args);

		CUDA_CPY_TO_HOST(memcd_host.key, memcd_dev.key, sizeof(granule_t)*nbSets*nbWays*KEY_SIZE);
		CUDA_CPY_TO_HOST(memcd_host.val, memcd_dev.val, sizeof(granule_t)*nbSets*nbWays*VAL_SIZE);
		CUDA_CPY_TO_HOST(memcd_host.state, memcd_dev.state, sizeof(granule_t)*nbSets*nbWays);
		CUDA_CPY_TO_HOST(memcd_host.ts, memcd_dev.ts, sizeof(granule_t)*nbSets*nbWays);

		storeArray("cache_state", nbSets*nbWays, sizeof(granule_t), memcd_host.state);
		storeArray("cache_ts_array", nbSets*nbWays, sizeof(granule_t), memcd_host.ts);
		storeArray("cache_keys", nbTXsT*nbBlks*nbThrs*KEY_SIZE, sizeof(granule_t), memcd_host.input_keys);
		storeArray("cache_vals", nbTXsT*nbBlks*nbThrs*VAL_SIZE, sizeof(granule_t), memcd_host.input_vals);
	} else {
		// load the others
		loadArray("cache_clock", 1, sizeof(int), memcd_host.curr_clock);
		loadArray("cache_ts_array", nbSets*nbWays, sizeof(granule_t), memcd_host.ts);
		loadArray("cache_keys", nbTXsT*nbBlks*nbThrs*KEY_SIZE, sizeof(granule_t), memcd_host.input_keys);
		loadArray("cache_vals", nbTXsT*nbBlks*nbThrs*VAL_SIZE, sizeof(granule_t), memcd_host.input_vals);

		CUDA_CPY_TO_DEV(memcd_dev.state, memcd_host.state, sizeof(granule_t)*nbSets*nbWays);
		CUDA_CPY_TO_DEV(memcd_dev.curr_clock, memcd_host.curr_clock, sizeof(int));
		CUDA_CPY_TO_DEV(memcd_dev.ts, memcd_host.ts, sizeof(int));
		CUDA_CPY_TO_DEV(memcd_dev.input_keys, memcd_host.input_keys, sizeof(granule_t)*nbTXsT*nbBlks*nbThrs*KEY_SIZE);
		CUDA_CPY_TO_DEV(memcd_dev.input_vals, memcd_host.input_vals, sizeof(granule_t)*nbTXsT*nbBlks*nbThrs*VAL_SIZE);
	}

	clock_gettime(CLOCK_REALTIME, &tTot1);
	while(nbReps-- > 0) {
		long mod = 0xFFFF;
		long rnd = RAND_R_FNC(seed) & mod;
		long probRead = prRead * 0xFFFF;

		clock_gettime(CLOCK_REALTIME, &t1);
		if (rnd > probRead) {
			// write kernel

			PR_prepare_noCallback(&args);
			memcdWriteTx<<<nbBlks, nbThrs>>>(args.dev, memcd_dev);
			PR_postrun_noCallback(&args);

		} else {
			// read kernel

			PR_prepare_noCallback(&args);
			memcdReadTx<<<nbBlks, nbThrs>>>(args.dev, memcd_dev);
			PR_postrun_noCallback(&args);
		}

		CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "last kernel sync");
		clock_gettime(CLOCK_REALTIME, &t2);
		*memcd_host.curr_clock += 1; // increments the batch clock
		CUDA_CPY_TO_DEV(memcd_dev.curr_clock, memcd_host.curr_clock, sizeof(int));
		nbIters++;
		tKernel_ms += (t2.tv_sec*1000.0 + t2.tv_nsec/1000000.0) - (t1.tv_sec*1000.0 + t1.tv_nsec/1000000.0);

		if (rnd <= probRead) {
			CUDA_CPY_TO_HOST(memcd_host.output, memcd_dev.output, sizeof(get_output_t)*nbTXsT*nbBlks*nbThrs);
			// check response
			for (int i = 0; i < nbTXsT*nbBlks*nbThrs; ++i) {
				if (memcd_host.output[i].isFound && memcd_host.output[i].val[0] != memcd_host.input_keys[i]) {
					printf("[Error] found wrong value!\n");
				}
				// if (!memcd_host.output[i].isFound) {
				// 	printf("key not found!\n");
				// }
			}
		}
	}
	CUDA_CPY_TO_HOST(memcd_host.curr_clock, memcd_dev.curr_clock, sizeof(int));
	storeArray("cache_clock", 1, sizeof(int), memcd_host.curr_clock);
	// ===========================

	// ===========================
	// Statistics
	clock_gettime(CLOCK_REALTIME, &tTot2);
	totT_ms += (tTot2.tv_sec*1000.0 + tTot2.tv_nsec/1000000.0) - (tTot1.tv_sec*1000.0 + tTot1.tv_nsec/1000000.0);

	printf("NB_BLOCKS\tNB_THREADS\tNB_TXsPerThr\tPROB_READ\tREPETITIONS\tTOT_TIME_MS\tTIME_IN_KERNEL_MS\n");
	printf("%i\t%i\t%i\t%f\t%i\t%f\t%f\n", nbBlks, nbThrs, nbTXsT, prRead, nbIters, totT_ms, tKernel_ms);
	// ===========================

	return EXIT_SUCCESS;
}

