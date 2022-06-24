// TODO: includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define PR_MAX_RWSET_SIZE 1024

#include "pr-stm.cuh"
#include "pr-stm-internal.cuh"

typedef struct times_
{
	long long int total;
	long long int runtime;
	long long int commit;
	long int nbReadOnly;
	long int nbUpdates;	
} time_rate;

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

__forceinline__ __device__ unsigned get_lane_id() {
	unsigned ret;
	asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
	return ret;
}

/*__global__ void memcdReadTx(PR_globalKernelArgs /* args , memcd_s input)
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
*/
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

__global__ void memcdTx(PR_globalKernelArgs /* args */, memcd_s input, uint64_t seed, float prRead, time_rate* times)
{
	long mod = 0xFFFF;
	long rnd;
	long probRead;// = prRead * 0xFFFF;

	int tid = PR_THREAD_IDX;
	long long int start_time_commit, stop_time_commit;
	long long int start_time_tx, stop_time_tx, start_time_total;;
	long int updates=0, reads=0;

	//int wid = tid >> 5;
	PR_enterKernel(tid);

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

	if(get_lane_id()==0)
	{
		rnd = RAND_R_FNC(seed) & mod;
	}
	rnd = __shfl_sync(0xffffffff, rnd, 0);
	probRead = prRead * 0xFFFF;

	for (int i = 0; i < nbTXsPerThread; ++i)
	{
		if(rnd > probRead)
		{
			//write kernel
			int input_key = input_keys[tid*nbTXsPerThread+i];
			int input_val = input_vals[tid*nbTXsPerThread+i];
			int target_set = HASH_TO_SET(input_key, nbSets);
			int possibleSlot = -1;
			int possibleSlotAge = -1;
			int possibleSlotState = -1;
			granule_t key;
			granule_t state;
			granule_t age;
			
			start_time_total = clock64();	
			PR_txBegin();
			start_time_tx = clock64();
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
			
			start_time_commit = clock64(); 
			PR_txCommit();
			stop_time_commit = clock64();
			stop_time_tx = clock64();

			times[tid].total   += stop_time_tx - start_time_total;
			times[tid].runtime += stop_time_tx - start_time_tx;
			times[tid].commit  += stop_time_commit - start_time_commit;
		}
		else
		{
			//read kernel
			get_output_t        *out = input.output;
			granule_t    *input_keys = input.input_keys;
			out[tid*nbTXsPerThread + i].isFound = 0;
			
			int input_key = input_keys[tid*nbTXsPerThread + i]; 
			int target_set = HASH_TO_SET(input_key, nbSets);
			
			start_time_total = clock64();
			PR_txBegin();
			start_time_tx = clock64();
			for (int j = 0; j < nbWays; ++j) {
				granule_t key = PR_read(&keys[j + target_set*nbWays]);
				granule_t state = PR_read(&states[j + target_set*nbWays]); // TODO: use the whole val
				if (key == input_key && (state&VALID)) { // found the key
					granule_t val = PR_read(&values[j + target_set*nbWays]); // TODO: use the whole val
					//PR_write(&timestamps[j + target_set*nbWays], curr_clock);
					// you could try read + normal write
					// ts = PR_read(&timestamps[thread_pos]); // assume read-before-write
					// timestamps[thread_pos] = ts;
					out[tid*nbTXsPerThread + i].isFound = 1;
					out[tid*nbTXsPerThread + i].val[0] = val; // TODO: value_size
					break;
				}
			}
			start_time_commit = clock64(); 
			PR_txCommit();
			stop_time_commit = clock64();
			stop_time_tx = clock64();

			times[tid].total   += stop_time_tx - start_time_total;
			times[tid].runtime += stop_time_tx - start_time_tx;
			times[tid].commit  += stop_time_commit - start_time_commit;
		}		
		if(rnd < probRead)
			reads++;
		else
			updates++;
	}
	times[tid].nbReadOnly += reads;
	times[tid].nbUpdates  += updates;
	PR_exitKernel();
}

void getKernelOutput(time_rate *h_times, uint threadNum, int peak_clk, float totT_ms, uint64_t nbCommits, uint64_t nbAborts, uint verbose)
{
  	double avg_total=0, avg_runtime=0, avg_commit=0;
  	long int totReads=0, totUpdates=0;
	
	//long int nbAborts = *PR_sumNbAborts;
	long int commits;

	for(int i=0; i<threadNum; i++)
	{
		if(h_times[i].runtime < 0) printf("T%d: %li\n", i, h_times[i].runtime);
		avg_total   += h_times[i].total;
		avg_runtime += h_times[i].runtime;
		avg_commit 	+= h_times[i].commit;

		totReads 	+= h_times[i].nbReadOnly;
		totUpdates	+= h_times[i].nbUpdates;
	}
	
	commits = totReads + totUpdates;
	long int denom = nbCommits*peak_clk;
	avg_total	/= denom;
	avg_runtime	/= denom;
	avg_commit 	/= denom;

	float rt_commit=0.0;
	rt_commit	=	avg_commit / avg_runtime;

	//printf("Commits: %d\t%d\n\n", commits, nbCommits);
	
	if(verbose)
		printf("AbortPercent\t%f %%\nThroughtput\t%f\n\nTotal\t\t%f\nRuntime\t\t%f\nCommit\t\t%f\t%.2f%%\n",
			(float)nbAborts/(nbAborts+nbCommits)*100.0,
			nbCommits/totT_ms*1000.0,
			avg_total,
			avg_runtime,
			avg_commit,
			rt_commit*100.0
			);
	else
		printf("%f\t%f\t%f\t%f\t%f\t%f\n", 
			(float)nbAborts/(nbAborts+nbCommits)*100.0,
			nbCommits/totT_ms*1000.0,
			avg_total,
			avg_runtime,
			avg_commit,
			rt_commit*100.0
			);
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
	  "  4) client config - nb blocks      \n"
	  "  5) client config - nb threads     \n"
	  "  6) nb TXs per thread              \n"
	  //"  8) nb repetitions                 \n"
	  "  7) prob read TX kernel            \n"
	  "  8) run duration                   \n"
	  "  9) verbose		                   \n"
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
	long   nbKeys = memcd_host.nbKeys         = memcd_host.nbSets*memcd_host.nbWays*2;	//(long)atol(argv[argCnt++]);
	int    nbBlks = memcd_host.nbBlocks       = (int)atol(argv[argCnt++]);
	int    nbThrs = memcd_host.nbThreads      = (int)atol(argv[argCnt++]);
	int    nbTXsT = memcd_host.nbTXsPerThread = (int)atol(argv[argCnt++]);
	float  prRead = memcd_host.probReadKernel = (float)atof(argv[argCnt++])/100.0;
	int    totalDuration                      = (int)atol(argv[argCnt++]);
	int    verbose                            = (int)atol(argv[argCnt++]);

	float tKernel_ms = 0.0, totT_ms = 0.0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int nbIters = 0;
	// ===========================

	// select GPU device
	CUDA_CHECK_ERROR(cudaSetDevice(0), "GPU select");

	int peak_clk=1;
	cudaError_t err = cudaDeviceGetAttribute(&peak_clk, cudaDevAttrClockRate, 0);
  	if (err != cudaSuccess) {printf("cuda err: %d at line %d\n", (int)err, __LINE__); return 1;}

  	time_rate *d_times, *h_times;

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
	CUDA_DUAL_ALLOC(h_times, d_times, sizeof(time_rate)*nbThrs*nbBlks);

	*memcd_host.curr_clock = 1;
	memset(memcd_host.state, 0, sizeof(granule_t)*nbSets*nbWays);
	memset(h_times, 0, sizeof(time_rate)*nbThrs*nbBlks);
	
	CUDA_CPY_TO_DEV(memcd_dev.state, memcd_host.state, sizeof(granule_t)*nbSets*nbWays);
	CUDA_CPY_TO_DEV(memcd_dev.curr_clock, memcd_host.curr_clock, sizeof(int));
	CUDA_CPY_TO_DEV(d_times, h_times, sizeof(time_rate)*nbThrs*nbBlks);
	// ===========================

	// ===========================
	// Main loop
	PR_init(1);
	pr_tx_args_s args;

	uint64_t *sumNbAborts;
	uint64_t *sumNbCommits;

	CUDA_CHECK_ERROR(cudaMallocManaged(&sumNbCommits, sizeof(uint64_t)), "Could not alloc");
	CUDA_CHECK_ERROR(cudaMallocManaged(&sumNbAborts, sizeof(uint64_t)), "Could not alloc");

	*sumNbAborts = 0;
	*sumNbCommits = 0;

	PR_blockNum = nbBlks;
	PR_threadNum = nbThrs;

	// gen keys
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

	// populates memory structs
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

	//PR_reduceCommitAborts<<<PR_blockNum, PR_threadNum, 0, PR_streams[PR_currentStream]>>>
	//	(0, PR_currentStream, args.dev, sumNbCommits, sumNbAborts);
	//CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "pop sync");
	//printf("BEFORE RESET\nCommits: %d\tAborts: %d\n", *sumNbCommits, *sumNbAborts);

	*sumNbAborts = 0;
	*sumNbCommits = 0;
	PR_resetStatistics(&args);

	//PR_reduceCommitAborts<<<PR_blockNum, PR_threadNum, 0, PR_streams[PR_currentStream]>>>
	//	(0, PR_currentStream, args.dev, sumNbCommits, sumNbAborts);
	//CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "pop sync");
	//printf("AFTER RESET\nCommits: %d\tAborts: %d\n", *sumNbCommits, *sumNbAborts);

	while(totT_ms < totalDuration*1000)
	{
		
		// regen keys
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
		// populates memory structs with new keys
		CUDA_CPY_TO_DEV(memcd_dev.input_keys, memcd_host.input_keys, sizeof(granule_t)*nbTXsT*nbBlks*nbThrs*KEY_SIZE);
		CUDA_CPY_TO_DEV(memcd_dev.input_vals, memcd_host.input_vals, sizeof(granule_t)*nbTXsT*nbBlks*nbThrs*VAL_SIZE);

		cudaEventRecord(start);
		PR_prepare_noCallback(&args);
		memcdTx<<<nbBlks, nbThrs>>>(args.dev, memcd_dev, seed, prRead, d_times);
		PR_postrun_noCallback(&args);
		cudaEventRecord(stop);

		CUDA_CHECK_ERROR(cudaEventSynchronize(stop), "in kernel");
		
		*memcd_host.curr_clock += 1; // increments the batch clock
		CUDA_CPY_TO_DEV(memcd_dev.curr_clock, memcd_host.curr_clock, sizeof(int));
		nbIters++;

		cudaEventElapsedTime(&tKernel_ms, start, stop);
		totT_ms += tKernel_ms;

		PR_reduceCommitAborts<<<PR_blockNum, PR_threadNum, 0, PR_streams[PR_currentStream]>>>
			(0, PR_currentStream, args.dev, sumNbCommits, sumNbAborts);
		CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "pop sync");
		//printf("%d\tCommits: %d\tAborts: %d\n", nbIters, *sumNbCommits, *sumNbAborts);
	}
	//PR_reduceCommitAborts<<<PR_blockNum, PR_threadNum, 0, PR_streams[PR_currentStream]>>>
	//	(0, PR_currentStream, args.dev, sumNbCommits, sumNbAborts);
	//CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "pop sync");
	//printf("FINAL\nCommits: %d\tAborts: %d\n", *sumNbCommits, *sumNbAborts);
	// ===========================

	// ===========================
	// Statistics
	//clock_gettime(CLOCK_REALTIME, &tTot2);
	//totT_ms += (tTot2.tv_sec*1000.0 + tTot2.tv_nsec/1000000.0) - (tTot1.tv_sec*1000.0 + tTot1.tv_nsec/1000000.0);

	//printf("NB_BLOCKS\tNB_THREADS\tNB_TXsPerThr\tPROB_READ\tREPETITIONS\tTOT_TIME_MS\tTIME_IN_KERNEL_MS\n");
	//printf("%i\t%i\t%i\t%f\t%i\t%f\t%f\n", nbBlks, nbThrs, nbTXsT, prRead, nbIters, totT_ms, tKernel_ms);
	// ===========================
	CUDA_CPY_TO_HOST(h_times, d_times, sizeof(time_rate)*nbThrs*nbBlks);
	getKernelOutput(h_times, nbBlks*nbThrs, peak_clk, totT_ms, *sumNbCommits, *sumNbAborts, verbose);

	return EXIT_SUCCESS;
}

