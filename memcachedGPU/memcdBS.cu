// TODO: includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <unistd.h>

#include "cuda_util.h"
#include "baseline/API.cuh"
//#include "keyGenAndCacheStore.h"

//#define USE_ZIPF      0
//#define ZIPF_PARAM  0.5 // TODO: this is not standard Zipf parameter

// TODO: I'm not making use of larger VAL/KEY sizes
#define VAL_SIZE   1
#define KEY_SIZE   1

#define THREAD_IDX       (threadIdx.x + blockIdx.x * blockDim.x)
#define KERNEL_DURATION 5

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
	((_hashTmp) % _nbSets); \
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
	VertionedDataItem *key;           /* keys in global memory */
	VertionedDataItem *val;           /* values in global memory */
	VertionedDataItem *ts;            /* last access TS in global memory */
	VertionedDataItem *state;         /* state in global memory */
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

//__device__ int waitMem;

// TODO: IMPORTANT ---> set PR_MAX_RWSET_SIZE to number of ways

__forceinline__ __device__ unsigned get_lane_id() {
	unsigned ret;
	asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
	return ret;
}


__global__ void memcdWriteTx(TXRecord* record, TMmetadata* metadata, memcd_s input, Statistics* stats, time_rate* times)
{
	int tid = THREAD_IDX;
	local_metadata txData;
	bool result;

	// TODO: blockDim.x must be multiple of num_ways --> else this does not work
	int nbSets = input.nbSets;
	int nbWays = input.nbWays;
	int nbTXsPerThread = input.nbTXsPerThread;

	VertionedDataItem       *keys = input.key;
	VertionedDataItem     *values = input.val;
	VertionedDataItem *timestamps = input.ts;
	VertionedDataItem     *states = input.state;
	//get_output_t        *out = input.output;
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
		do
		{
			TXBegin(*metadata, &txData);
			for (int j = 0; j < nbWays; ++j) {
				key = 	TXRead(&keys[j + target_set*nbWays], &txData);
				state = TXRead(&states[j + target_set*nbWays], &txData);
				age = 	TXRead(&timestamps[j + target_set*nbWays], &txData);
				if ((possibleSlot == -1) // init
						|| (!(possibleSlotState&INVALID) && age < possibleSlotAge) // empty slot
						|| (key == input_key && (state&VALID)) // found key slot
						) {
					possibleSlot = j;
					possibleSlotState = state;
					possibleSlotAge = age;
				}
			}
			TXWrite(&keys[possibleSlot + target_set*nbWays], input_key, &txData);
			TXWrite(&values[possibleSlot + target_set*nbWays], input_val, &txData);
			TXWrite(&timestamps[possibleSlot + target_set*nbWays], curr_clock, &txData);
			TXWrite(&states[possibleSlot + target_set*nbWays], VALID, &txData);
			
			result=TXCommit(tid, record, metadata, txData, stats, times);
		}while(!result);
	}
}

__global__ void memcdTx(TXRecord* record, TMmetadata* metadata, memcd_s input, uint64_t seed, float prRead, Statistics* stats, time_rate* times)
{
	long mod = 0xFFFF;
	long rnd;
	long probRead;// = prRead * 0xFFFF;


	local_metadata txData;
	bool result;

	int tid = THREAD_IDX;
	uint64_t state = seed+tid;
	//int wid = tid >> 5;

	//profile metrics
	long long int start_time_commit, stop_time_commit;
	long long int start_time_tx, stop_time_tx, start_time_total;

	int nbSets = input.nbSets;
	int nbWays = input.nbWays;
	int nbTXsPerThread = input.nbTXsPerThread;

	VertionedDataItem       *keys = input.key;
	VertionedDataItem     *values = input.val;
	VertionedDataItem *timestamps = input.ts;
	VertionedDataItem     *states = input.state;
	int           curr_clock = *((int*)input.curr_clock);
	int          *input_keys = input.input_keys;
	int          *input_vals = input.input_vals;

	for (int i = 0; i < nbTXsPerThread; ++i)
	{
		///////
		//decide whether the warp will be do update or read-only set of txs
		if(get_lane_id()==0)
		{
			rnd = RAND_R_FNC(state) & mod;
		}
		rnd = __shfl_sync(0xffffffff, rnd, 0);
		probRead = prRead * 0xFFFF;
		///////

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
			do
			{
				start_time_total = clock64();
				TXBegin(*metadata, &txData);
				start_time_tx = clock64();
				for (int j = 0; j < nbWays; ++j) {
					key = 	TXRead(&keys[j + target_set*nbWays], &txData);
					state = TXRead(&states[j + target_set*nbWays], &txData);
					age = 	TXRead(&timestamps[j + target_set*nbWays], &txData);
					if(key == input_key && (state&VALID)){
						possibleSlot = j;
						possibleSlotState = state;
						possibleSlotAge = age;
						break;
					}
					if ((possibleSlot == -1) // init
							|| (!(possibleSlotState&INVALID) && age < possibleSlotAge) // empty slot
							) {
						possibleSlot = j;
						possibleSlotState = state;
						possibleSlotAge = age;
					}
				}
				if(txData.isAborted==true){ atomicAdd(&(stats->nbAbortsDataAge), 1); continue;}

				TXWrite(&keys[possibleSlot + target_set*nbWays], input_key, &txData);
				TXWrite(&values[possibleSlot + target_set*nbWays], input_val, &txData);
				TXWrite(&timestamps[possibleSlot + target_set*nbWays], curr_clock, &txData);
				TXWrite(&states[possibleSlot + target_set*nbWays], VALID, &txData);
				
				start_time_commit = clock64();
				result=TXCommit(tid, record, metadata, txData, stats, times);
				stop_time_commit = clock64();
				if(result)
					atomicAdd(&(stats->nbCommits), 1);

			}while(!result);
			stop_time_tx = clock64();

			times[tid].total   += stop_time_tx - start_time_total;
			times[tid].runtime += stop_time_tx - start_time_tx;
			times[tid].commit += stop_time_commit - start_time_commit;
		}
		else
		{
			//read kernel
			get_output_t        *out = input.output;
			granule_t    *input_keys = input.input_keys;
			out[tid*nbTXsPerThread + i].isFound = 0;
			
			int input_key = input_keys[tid*nbTXsPerThread + i]; 
			int target_set = HASH_TO_SET(input_key, nbSets);
			
			do
			{
				start_time_total = clock64();
				TXBegin(*metadata, &txData);
				start_time_tx = clock64();
				for (int j = 0; j < nbWays; ++j) {
					granule_t key =   TXReadOnly(&keys[j + target_set*nbWays], &txData);
					granule_t state = TXReadOnly(&states[j + target_set*nbWays], &txData);
					if (key == input_key && (state&VALID)) { // found the key
						granule_t val = TXReadOnly(&values[j + target_set*nbWays], &txData); // TODO: use the whole val
						out[tid*nbTXsPerThread + i].isFound = 1;
						out[tid*nbTXsPerThread + i].val[0] = val; // TODO: value_size
						break;
					}
				}
				if(txData.isAborted==true){ atomicAdd(&(stats->nbAbortsDataAge), 1); continue;}

				start_time_commit = clock64();
				result=TXCommit(tid, record, metadata, txData, stats, times);
				stop_time_commit = clock64();
				if(result)
					atomicAdd(&(stats->nbCommits), 1);

			}while(!result);
			stop_time_tx = clock64();

			times[tid].total   += stop_time_tx - start_time_total;
			times[tid].runtime += stop_time_tx - start_time_tx;
			times[tid].commit  += stop_time_commit - start_time_commit;
		}
	}
}

void getKernelOutput(Statistics *h_stats, time_rate *h_times, uint threadNum, int peak_clk, float totT_ms, int nbIters, int verbose)
{
	double avg_total=0, avg_runtime=0, avg_commit=0, avg_wb=0, avg_val1=0, avg_val2=0, avg_rwb=0, avg_comp=0;
	long int totUpdates=0, totReads=0;
	for(int i=0; i<threadNum; i++)
	{
		if(h_times[i].runtime < 0) printf("T%d: %li\n", i, h_times[i].runtime);
		avg_total   += h_times[i].total;
		avg_runtime += h_times[i].runtime;
		avg_commit 	+= h_times[i].commit;
		avg_wb 		+= h_times[i].dataWrite;
		avg_val1	+= h_times[i].val1;
		avg_val2	+= h_times[i].val2;
		avg_rwb		+= h_times[i].recordWrite;
		avg_comp 	+= h_times[i].comparisons;
	
		totUpdates 	+= h_times[i].nbUpdates;
		totReads	+= h_times[i].nbReadOnly;
	}
	
	long int denom = (long)h_stats->nbCommits*peak_clk;
	avg_total	/= denom;
	avg_runtime	/= denom;
	avg_commit 	/= denom;
	avg_wb 		/= denom;
	avg_val1 	/= denom;
	avg_val2 	/= denom;
	avg_rwb 	/= denom;
	avg_comp	/= h_stats->nbCommits;

	float rt_commit=0.0, rt_wb=0.0, rt_val1=0.0, rt_val2=0.0, rt_rwb=0.0, dummy=0.0;
	rt_commit	=	avg_commit / avg_runtime;
	rt_wb	 	=	avg_wb / avg_runtime;
	rt_val1	 	=	avg_val1 / avg_runtime;
	rt_val2	 	=	avg_val2 / avg_runtime;
	rt_rwb	 	=	avg_rwb / avg_runtime;

	int nbAborts = h_stats->nbAbortsDataAge + h_stats->nbAbortsRecordAge + h_stats->nbAbortsReadWrite;

	if(verbose)
		printf("AbortPercent\t%f %%\nThroughtput\t%f\n\nAbortDataAge\t%f %%\nAbortRecAge\t%f %%\nAbortReadWrite\t%f %%\nAbortPreVal\t%f %%\n\nTotal\t\t%f\nRuntime\t\t%f\nCommit\t\t%f\t%.2f%%\nWaitTime\t%f\t%.2f%%\nPreValidation\t%f\t%.2f%%\n1stValidation\t%f\t%.2f%%\nRecInsertVals\t%f\t%.2f%%\nRecInsert\t%f\t%.2f%%\nWriteBack\t%f\t%.2f%%\n\nComparisons\t%f\nTotalIters\t%d\n", 
			(float)nbAborts/(nbAborts+h_stats->nbCommits)*100.0,
			h_stats->nbCommits/totT_ms*1000.0,
			(float)h_stats->nbAbortsDataAge/(nbAborts+h_stats->nbCommits)*100.0,
			(float)h_stats->nbAbortsRecordAge/(nbAborts+h_stats->nbCommits)*100.0,
			(float)h_stats->nbAbortsReadWrite/(nbAborts+h_stats->nbCommits)*100.0,
			(float)h_stats->nbAbortsWriteWrite/(nbAborts+h_stats->nbCommits)*100.0,
			avg_total,
			avg_runtime,
			avg_commit,
			rt_commit*100.0,
			dummy,
			dummy,
			dummy,
			dummy,
			avg_val1,
			rt_val1*100.0,
			avg_val2,
			rt_val2*100.0,
			avg_rwb,
			rt_rwb*100.0,
			avg_wb,
			rt_wb*100.0,
			avg_comp,
			nbIters
			);
	else
		printf("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%d\n", 
			(float)nbAborts/(nbAborts+h_stats->nbCommits)*100.0,
			h_stats->nbCommits/totT_ms*1000.0,
			(float)h_stats->nbAbortsDataAge/(nbAborts+h_stats->nbCommits)*100.0,
			(float)h_stats->nbAbortsRecordAge/(nbAborts+h_stats->nbCommits)*100.0,
			(float)h_stats->nbAbortsReadWrite/(nbAborts+h_stats->nbCommits)*100.0,
			(float)h_stats->nbAbortsWriteWrite/(nbAborts+h_stats->nbCommits)*100.0,
			avg_total,
			avg_runtime,
			avg_commit,
			rt_commit*100.0,
			dummy,
			dummy,
			dummy,
			dummy,
			avg_val1,
			rt_val1*100.0,
			avg_val2,
			rt_val2*100.0,
			avg_rwb,
			rt_rwb*100.0,
			avg_wb,
			rt_wb*100.0,
			avg_comp,
			nbIters
			);
}

/*
static void fillInputs(int idx, int total, long long item) {
	for (int j = 0; j < KEY_SIZE; ++j) {
		memcd_host.input_keys[idx + j*total] = (int)item;
	}
	for (int j = 0; j < VAL_SIZE; ++j) {
		memcd_host.input_vals[idx + j*total] = (int)item;
	}
}*/

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

	TXRecord* d_records;
	TMmetadata* d_metadata;
	Statistics *d_stats, *h_stats;
	time_rate *d_times, *h_times;

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

	// ===========================
	// memory setup
	memcpy(&memcd_dev, &memcd_host, sizeof(memcd_dev));

	CUDA_DUAL_ALLOC(memcd_host.key, memcd_dev.key, sizeof(VertionedDataItem)*nbSets*nbWays*KEY_SIZE);
	CUDA_DUAL_ALLOC(memcd_host.val, memcd_dev.val, sizeof(VertionedDataItem)*nbSets*nbWays*VAL_SIZE);
	CUDA_DUAL_ALLOC(memcd_host.state, memcd_dev.state, sizeof(VertionedDataItem)*nbSets*nbWays);
	CUDA_DUAL_ALLOC(memcd_host.ts, memcd_dev.ts, sizeof(VertionedDataItem)*nbSets*nbWays);
	CUDA_DUAL_ALLOC(memcd_host.curr_clock, memcd_dev.curr_clock, sizeof(int));
	CUDA_DUAL_ALLOC(memcd_host.input_keys, memcd_dev.input_keys, sizeof(granule_t)*nbThrs*nbBlks*nbTXsT*KEY_SIZE);
	CUDA_DUAL_ALLOC(memcd_host.input_vals, memcd_dev.input_vals, sizeof(granule_t)*nbThrs*nbBlks*nbTXsT*VAL_SIZE);
	CUDA_DUAL_ALLOC(memcd_host.output, memcd_dev.output, sizeof(get_output_t)*nbThrs*nbBlks*nbTXsT);
	CUDA_DUAL_ALLOC(h_stats, d_stats, sizeof(Statistics));
	CUDA_DUAL_ALLOC(h_times, d_times, sizeof(time_rate)*nbThrs*nbBlks);

	*memcd_host.curr_clock = 1;
	memset(memcd_host.state, 0, sizeof(VertionedDataItem)*nbSets*nbWays);
	memset(h_stats, 0, sizeof(Statistics));
	memset(h_times, 0, sizeof(time_rate)*nbThrs*nbBlks);

	for(int i=0; i<nbSets*nbWays; i++)
	{
		memcd_host.key[i].head_ptr=1;
		memcd_host.val[i].head_ptr=1;
		memcd_host.state[i].head_ptr=1;
		memcd_host.ts[i].head_ptr=1;
	}

	CUDA_CPY_TO_DEV(memcd_dev.state, memcd_host.state, sizeof(VertionedDataItem)*nbSets*nbWays);
	CUDA_CPY_TO_DEV(memcd_dev.curr_clock, memcd_host.curr_clock, sizeof(int));
	CUDA_CPY_TO_DEV(d_stats, h_stats, sizeof(Statistics));
	CUDA_CPY_TO_DEV(d_times, h_times, sizeof(time_rate)*nbThrs*nbBlks);
	// ===========================

	// ===========================
	// Main loop
	TXInit(&d_records, &d_metadata);

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
	//memcdWriteTx<<<nbBlks, nbThrs>>>(d_records, d_metadata, memcd_dev, d_stats, d_times);
	//CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "pop sync");
	
	CUDA_CPY_TO_HOST(memcd_host.key, memcd_dev.key, sizeof(granule_t)*nbSets*nbWays*KEY_SIZE);
	CUDA_CPY_TO_HOST(memcd_host.val, memcd_dev.val, sizeof(granule_t)*nbSets*nbWays*VAL_SIZE);
	CUDA_CPY_TO_HOST(memcd_host.state, memcd_dev.state, sizeof(granule_t)*nbSets*nbWays);
	CUDA_CPY_TO_HOST(memcd_host.ts, memcd_dev.ts, sizeof(granule_t)*nbSets*nbWays);
	
	//reset performance metrics
	CUDA_CPY_TO_DEV(d_stats, h_stats, sizeof(Statistics));
	CUDA_CPY_TO_DEV(d_times, h_times, sizeof(time_rate)*nbThrs*nbBlks);

	while(totT_ms < totalDuration*1000)
	{
		// regen keys
		for (int i = 0; i < nbTXsT*nbBlks*nbThrs; ++i) {
			granule_t key = RAND_R_FNC(seed) % nbKeys;
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
		memcdTx<<<nbBlks, nbThrs>>>(d_records, d_metadata, memcd_dev, seed, prRead, d_stats, d_times);
		cudaEventRecord(stop);

		CUDA_CHECK_ERROR(cudaEventSynchronize(stop), "in kernel");

		*memcd_host.curr_clock += 1; // increments the batch clock
		CUDA_CPY_TO_DEV(memcd_dev.curr_clock, memcd_host.curr_clock, sizeof(int));
		nbIters++;

		cudaEventElapsedTime(&tKernel_ms, start, stop);
		totT_ms += tKernel_ms;

	}
	CUDA_CPY_TO_HOST(h_stats, d_stats, sizeof(Statistics));
	CUDA_CPY_TO_HOST(h_times, d_times, sizeof(time_rate)*nbThrs*nbBlks);

	getKernelOutput(h_stats, h_times, nbBlks*nbThrs, peak_clk, totT_ms, nbIters, verbose);

	return EXIT_SUCCESS;
}

