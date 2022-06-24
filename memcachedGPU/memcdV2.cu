// TODO: includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define PR_MAX_RWSET_SIZE 1024

#include "pr-stm.cuh"
#include "pr-stm-internal.cuh"
#include "keyGenAndCacheStore.h"

#define USE_ZIPF      1
#define ZIPF_PARAM  0.5 // TODO: this is not standard Zipf parameter

// TODO: I'm not making use of larger VAL/KEY sizes
#define VAL_SIZE   1
#define KEY_SIZE   1

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

#define HASH_TO_SET(_valueToHash, _nbSets) ({ \
	volatile uint64_t _hashTmp = (uint64_t)_valueToHash; \
	((RAND_R_FNC(_hashTmp)) % _nbSets); \
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
  VALID       = 1
};

// ===========================
// global info of benchmark
memcd_s memcd_host, memcd_dev;
// ===========================

// TODO: IMPORTANT ---> set PR_MAX_RWSET_SIZE to number of ways

__global__ void memcdReadTx(PR_globalKernelArgs /* args */, memcd_s input)
{
	int tid = PR_THREAD_IDX;
	PR_enterKernel(tid);

	int i, j;
	int nbSets = input.nbSets;
	int nbWays = input.nbWays;
	int nbTXsPerThread = input.nbTXsPerThread;

	granule_t       *keys = input.key;
	granule_t     *values = input.val;
	granule_t *timestamps = input.ts;
	granule_t     *states = input.state;

	get_output_t        *out = input.output;
	int           curr_clock = *((int*)input.curr_clock);
	granule_t    *input_keys = input.input_keys;

	for (int i = 0; i < nbTXsPerThread; ++i) { // num_ways keys
		out[tid*nbTXsPerThread + i].isFound = 0;
	}

	for (i = 0; i < nbTXsPerThread; ++i) {
		int input_key = input_keys[tid*nbTXsPerThread + i]; 
		int target_set = HASH_TO_SET(input_key, nbSets);
		
		PR_txBegin();
		for (j = 0; j < nbWays; ++j) {
			granule_t key = PR_read(&keys[j + target_set*nbWays]);
			granule_t state = PR_read(&states[j + target_set*nbWays]); // TODO: use the whole val
			if (key == input_key && (state&VALID)) { // found the key
				granule_t val = PR_read(&values[j + target_set*nbWays]); // TODO: use the whole val
				PR_write(&timestamps[j + target_set*nbWays], curr_clock);
				// you could try read + normal write
				// ts = PR_read(&timestamps[thread_pos]); // assume read-before-write
				// timestamps[thread_pos] = ts;
				out[tid*nbTXsPerThread + i].isFound = 1;
				out[tid*nbTXsPerThread + i].val[0] = val; // TODO: value_size
				break;
			}
		}
		PR_txCommit();
	}
	PR_exitKernel();
}

__global__ void memcdWriteTx(PR_globalKernelArgs /* args */, memcd_s input)
{
	int tid = PR_THREAD_IDX;
	PR_enterKernel(tid);

	// TODO: blockDim.x must be multiple of num_ways --> else this does not work
	int nbSets = input.nbSets;
	int nbWays = input.nbWays;
	int nbTXsPerThread = input.nbTXsPerThread;

	granule_t       *keys = input.key;
	granule_t     *values = input.val;
	granule_t *timestamps = input.ts;
	granule_t     *states = input.state;
	int           curr_clock = *((int*)input.curr_clock);
	int          *input_keys = input.input_keys;
	int          *input_vals = input.input_vals;

	for (int i = 0; i < nbTXsPerThread; ++i) {
		int input_key = input_keys[tid*nbTXsPerThread + i];
		int input_val = input_vals[tid*nbTXsPerThread + i];
		int target_set = HASH_TO_SET(input_key, nbSets);
		int possibleSlot = -1;
		int possibleSlotAge = -1;
		int possibleSlotState = -1;
		granule_t key;
		granule_t state;
		granule_t age;
		
		PR_txBegin();
		for (int j = 0; j < nbWays; ++j) {
			key = PR_read(&keys[j + target_set*nbWays]);
			state = PR_read(&states[j + target_set*nbWays]); // TODO: use the whole val
			age = PR_read(&timestamps[j + target_set*nbWays]); // TODO: use the whole val
			if ((possibleSlot == -1) // init
					|| (!(possibleSlotState&INVALID) && age < possibleSlotAge) // empty slot
					|| (key == input_key && (state&VALID)) // found key slot
					) {
				possibleSlot = j;
				possibleSlotState = state;
				possibleSlotAge = age;
			}
		}
		PR_write(&keys[possibleSlot + target_set*nbWays], input_key);
		PR_write(&values[possibleSlot + target_set*nbWays], input_val); // TODO: use the whole val
		PR_write(&timestamps[possibleSlot + target_set*nbWays], curr_clock);
		PR_write(&states[possibleSlot + target_set*nbWays], VALID);
		PR_txCommit();
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

	uint64_t seed = memcd_host.seed           = (long)atol(argv[argCnt++]);
	int    nbSets = memcd_host.nbSets         = (int)atol(argv[argCnt++]);
	int    nbWays = memcd_host.nbWays         = (int)atol(argv[argCnt++]);
	long   nbKeys = memcd_host.nbKeys         = (long)atol(argv[argCnt++]);
	int    nbBlks = memcd_host.nbBlocks       = (int)atol(argv[argCnt++]);
	int    nbThrs = memcd_host.nbThreads      = (int)atol(argv[argCnt++]);
	int    nbTXsT = memcd_host.nbTXsPerThread = (int)atol(argv[argCnt++]);
	float  prRead = memcd_host.probReadKernel = (float)atof(argv[argCnt++]);
	int    nbReps = memcd_host.nbReps         = (int)atol(argv[argCnt++]);

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

	// gen keys
#ifdef USE_ZIPF
	fill_array_with_items(nbTXsT*nbBlks*nbThrs, nbKeys, ZIPF_PARAM, /* TODO: implement me */fillInputs);
#else
	for (int i = 0; i < nbTXsT*nbBlks*nbThrs; ++i) {
		granule_t key = RAND_R_FNC(seed) % nbKeys;
		// if (key == 0) printf("key is 0\n");
		for (int j = 0; j < KEY_SIZE; ++j) {
			memcd_host.input_keys[i + j*nbTXsT*nbBlks*nbThrs] = key;
		}
		for (int j = 0; j < VAL_SIZE; ++j) {
			memcd_host.input_vals[i + j*nbTXsT*nbBlks*nbThrs] = key;
		}
	}
#endif

	// populates memory structs
	CUDA_CPY_TO_DEV(memcd_dev.state, memcd_host.state, sizeof(granule_t)*nbSets*nbWays);
	CUDA_CPY_TO_DEV(memcd_dev.curr_clock, memcd_host.curr_clock, sizeof(int));
	CUDA_CPY_TO_DEV(memcd_dev.input_keys, memcd_host.input_keys, sizeof(granule_t)*nbTXsT*nbBlks*nbThrs*KEY_SIZE);
	CUDA_CPY_TO_DEV(memcd_dev.input_vals, memcd_host.input_vals, sizeof(granule_t)*nbTXsT*nbBlks*nbThrs*VAL_SIZE);
	CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "mem sync");

	// Populates the cache
	PR_prepare_noCallback(&args);
	memcdWriteTx<<<nbBlks, nbThrs>>>(args.dev, memcd_dev);
	PR_postrun_noCallback(&args);
	CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "pop sync");

	CUDA_CPY_TO_HOST(memcd_host.key, memcd_dev.key, sizeof(granule_t)*nbSets*nbWays*KEY_SIZE);
	CUDA_CPY_TO_HOST(memcd_host.val, memcd_dev.val, sizeof(granule_t)*nbSets*nbWays*VAL_SIZE);
	CUDA_CPY_TO_HOST(memcd_host.state, memcd_dev.state, sizeof(granule_t)*nbSets*nbWays);
	CUDA_CPY_TO_HOST(memcd_host.ts, memcd_dev.ts, sizeof(granule_t)*nbSets*nbWays);

	clock_gettime(CLOCK_REALTIME, &tTot1);
	while(nbReps-- > 0) {
		long mod = 0xFFFF;
		long rnd = RAND_R_FNC(seed) & mod;
		long probRead = prRead * 0xFFFF;

		// regen keys
#ifdef USE_ZIPF
		zipf_setup(nbKeys, ZIPF_PARAM);
		for (int i = 0; i < nbTXsT*nbBlks*nbThrs; ++i) {
			granule_t key = zipf_gen();
#else
		for (int i = 0; i < nbTXsT*nbBlks*nbThrs; ++i) {
			granule_t key = RAND_R_FNC(seed) % nbKeys;
#endif
			for (int j = 0; j < KEY_SIZE; ++j) {
				memcd_host.input_keys[i + j*nbTXsT*nbBlks*nbThrs] = key;
			}
			for (int j = 0; j < VAL_SIZE; ++j) {
				memcd_host.input_vals[i + j*nbTXsT*nbBlks*nbThrs] = key;
			}
		}

		// populates memory structs with new keys
		CUDA_CPY_TO_DEV(memcd_dev.input_keys, memcd_host.input_keys, sizeof(granule_t)*nbTXsT*nbBlks*nbThrs*KEY_SIZE);
		CUDA_CPY_TO_DEV(memcd_dev.input_vals, memcd_host.input_vals, sizeof(granule_t)*nbTXsT*nbBlks*nbThrs*VAL_SIZE);

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
					printf("[Error] found wrong value! KEY=%i VAL=%i\n",
						memcd_host.input_keys[i], memcd_host.output[i].val[0]);
					break;
				}
				// if (!memcd_host.output[i].isFound) {
				// 	printf("key not found!\n");
				// }
			}
		}
	}
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

