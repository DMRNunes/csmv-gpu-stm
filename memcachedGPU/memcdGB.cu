// TODO: includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <time.h>
#include <fstream>
#include <string>
#include <iostream>
#include <limits>
#include <string.h>
#include <unistd.h>
#include <cassert>
#include <inttypes.h>
#include <unistd.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <cassert>
#include <inttypes.h>
#include <math.h>

#define SENDER_VERSION BASE_SEND

#include "sync_lib/common.h"
#include "cuda_util.h"
#include "mmcd_gb/API.cuh"
//#include "pr-stm-internal.cuh"

/*
 * app specific config
 */
////////////////////
#define NUM_RECEIVER 1
#define MSG_SIZE_OFFLOAD 2
#define MSG_SIZE_MAX MSG_SIZE_OFFLOAD
///////////////////

#define ALG_ARG_DEF uint* token
#define ALG_ARG  token
#define PRINT_DEBUG_ 1

#define SERV_ARG_DEF TMmetadata* metadata, TXRecord* records, readSet* rs, writeSet* ws, warpResult* wRes, Statistics* stats, time_rate* times
#define SERV_ARG metadata, records, rs, ws, wRes, stats, times

#define PERF_METRICS 1

//#define PR_MAX_RWSET_SIZE 1024

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

__device__ __forceinline__ void critcal_section(SERV_ARG_DEF, uint val0, int val1)
{

	uint tid = val0;
	uint timestamp = val1;
	int result=0;
	//if(get_lane_id()==0) printf("\t\t\tS%d: recv %d %d\n", thread_id_x()/32, tid/32, timestamp);
	result=TXAddToRecord(metadata, records, rs, ws, stats, times, timestamp, tid);
	//if(get_lane_id()==0) printf("\t\t\tS%d: sent %d %d\n", thread_id_x()/32, tid/32, result);
	
	wRes[tid/32].lane_result[tid%32]=result;
	__threadfence();
	if(get_lane_id() == 0)
		wRes[tid/32].valid_entry=1;

}

#include "sync_lib/one_phase_def.h"
#include "sync_lib/msg_config.h"
#include "sync_lib/msg_aux.h"
#include "sync_lib/msg_passing.h"
#include "sync_lib/one_phase_server.h"

__device__ bool TXCommit(gbc_t gbc, int tid, readSet* rs, writeSet* ws, uint timestamp, warpResult* wRes, uint retry, Statistics* stats, time_rate* times)
{
	/////////////////////////////
	//Commit process
	/////////////////////////////
	uint wid = tid/32;

	bool readOnly = ws[tid].size==0? true: false;
	uint saved_write_ptr = NOT_FOUND;
	uint valid_msg=1;
	uint dst = 0;
	uint val0 = tid;
	int val1;
	uint result;
	bool isAborted;

	//profile metrics
	long long int start_writeback=0, stop_writeback=0;
	long long int start_wait=0, stop_wait=0;
	long long int start_preVal=0, stop_preVal=0;

	//check if all the transactions are read only
	if(vote_ballot(readOnly))
	{
		start_preVal 	= stop_preVal 	 = clock64();
		start_wait		= stop_wait 	 = clock64();
		start_writeback = stop_writeback = clock64();
		atomicAdd(&(stats->nbCommits), 1);
		retry=0;
	}
	else
	{
		if(retry==1) start_preVal = clock64();
		isAborted = !TXPreValidation(tid, rs, ws);
		if(retry==1) stop_preVal = clock64();

		if((retry==0) || readOnly)
			val1=-1;
		else if( isAborted )
		{
			atomicAdd(&(stats->nbAbortsPreValid), 1);
			val1=-1;
		}
		else
			val1=timestamp;
		if(retry==1) start_wait = clock64();
		if(get_lane_id()==0)
			saved_write_ptr = atomicAdd(&(gbc.write_ptr[dst]), warpSize);
		saved_write_ptr = shuffle_idx(saved_write_ptr, 0) + get_lane_id();
		do
		{
			if(base_send(gbc, dst, valid_msg, saved_write_ptr, _offload_tail_ptr_cpy, SEND_ARG))
				valid_msg = 0;
		}
		while(vote_ballot(valid_msg) != 0);
		if(get_lane_id()==0)
		{
			while(wRes[wid].valid_entry==0);
		}
		result = wRes[wid].lane_result[get_lane_id()];
		if(retry==1) stop_wait = clock64();

		if(result != 0)
		{
			start_writeback = clock64();
			TXWriteBack(result, ws[tid]);
			stop_writeback = clock64();
			atomicAdd(&(stats->nbCommits), 1);
			retry=0;
			times[tid].dataWrite+= stop_writeback - start_writeback;
			times[tid].wait 	+= stop_wait - start_wait;
			times[tid].preValidation += stop_preVal - start_preVal;
		}
		//reset scoreboard
		if(get_lane_id()==0)
			wRes[wid].valid_entry=0;
	}
	return retry;
}

////////////////////////
////////////////////////
////////////////////////

__global__ void memcdWriteTx(gbc_t gbc, memcd_s input, readSet* rs, writeSet* ws, warpResult* wRes, Statistics* stats, time_rate* times)
{
	init_base(_offload_tail_ptr_cpy);
	__syncthreads();

	int tid = THREAD_IDX;
	int retry=1;
	uint timestamp;
	bool isAborted;

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
			timestamp=TXBegin(tid, ws, rs);
			if(retry)
			{
				for (int j = 0; j < nbWays && isAborted==false; ++j) {
					key = 	TXRead(&keys[j + target_set*nbWays], timestamp, rs, ws, tid, &isAborted);
					state = TXRead(&states[j + target_set*nbWays], timestamp, rs, ws, tid, &isAborted);
					age = 	TXRead(&timestamps[j + target_set*nbWays], timestamp, rs, ws, tid, &isAborted);
					if ((possibleSlot == -1) // init
							|| (!(possibleSlotState&INVALID) && age < possibleSlotAge) // empty slot
							|| (key == input_key && (state&VALID)) // found key slot
							) {
						possibleSlot = j;
						possibleSlotState = state;
						possibleSlotAge = age;
					}
				}
				TXWrite(&keys[possibleSlot + target_set*nbWays], input_key, ws, tid);
				TXWrite(&values[possibleSlot + target_set*nbWays], input_val, ws, tid);
				TXWrite(&timestamps[possibleSlot + target_set*nbWays], curr_clock, ws, tid);
				TXWrite(&states[possibleSlot + target_set*nbWays], VALID, ws, tid);
			}
			retry=TXCommit(gbc, tid, rs, ws, timestamp, wRes, retry, stats, times);
		}while(vote_ballot(retry) != 0);
	}
}

__global__ void memcdTx(gbc_t gbc, memcd_s input, readSet* rs, writeSet* ws, warpResult* wRes, uint64_t seed, float prRead, Statistics* stats, time_rate* times)
{
	long mod = 0xFFFF;
	long rnd;
	long probRead;// = prRead * 0xFFFF;

	init_base(_offload_tail_ptr_cpy);
	__syncthreads();
	int retry=1, oldretry=1;
	uint timestamp;
	bool isAborted;



	//profile metrics

	long long int start_time_commit, stop_time_commit;
	long long int start_time_tx, stop_time_tx;

	int tid = THREAD_IDX;//, wid = tid >> 5;
	uint64_t state = seed+tid;
	//int wid = tid >> 5;

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
			int input_key = input_keys[tid*nbTXsPerThread];
			int input_val = input_vals[tid*nbTXsPerThread];
			int target_set = HASH_TO_SET(input_key, nbSets);
			int possibleSlot = -1;
			int possibleSlotAge = -1;
			int possibleSlotState = -1;
			granule_t key;
			granule_t state;
			granule_t age;

			retry=oldretry=1;
			do
			{
				if(retry==1) start_time_tx = clock64();
				timestamp=TXBegin(tid, ws, rs);
				isAborted=false;
				if(retry==1)
				{
					for (int j = 0; j < nbWays && isAborted==false; ++j) {
						key = 	TXRead(&keys[j + target_set*nbWays], timestamp, rs, ws, tid, &isAborted);
						state = TXRead(&states[j + target_set*nbWays], timestamp, rs, ws, tid, &isAborted);
						age = 	TXRead(&timestamps[j + target_set*nbWays], timestamp, rs, ws, tid, &isAborted);
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
					if(isAborted==true){ atomicAdd(&(stats->nbAbortsDataAge), 1); continue;}
					TXWrite(&keys[possibleSlot + target_set*nbWays], input_key, ws, tid);
					TXWrite(&values[possibleSlot + target_set*nbWays], input_val, ws, tid);
					TXWrite(&timestamps[possibleSlot + target_set*nbWays], curr_clock, ws, tid);
					TXWrite(&states[possibleSlot + target_set*nbWays], VALID, ws, tid);
				}
				if(retry==1) start_time_commit = clock64();
				retry=TXCommit(gbc, tid, rs, ws, timestamp, wRes, retry, stats, times);
				if(retry!=oldretry) 
				{
					stop_time_commit = clock64();
					stop_time_tx = clock64();
					oldretry=retry;
				}
			}while(vote_ballot(retry) != 0);

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
			
			retry=oldretry=1;
			do
			{
				if(retry==1) start_time_tx = clock64();
				timestamp=TXBegin(tid, ws, rs);
				isAborted=false;
				for (int j = 0; j < nbWays && isAborted==false; ++j) {
					granule_t key =   TXRead(&keys[j + target_set*nbWays], timestamp, rs, ws, tid, &isAborted);
					granule_t state = TXRead(&states[j + target_set*nbWays], timestamp, rs, ws, tid, &isAborted);
					if (key == input_key && (state&VALID)) { // found the key
						granule_t val = TXRead(&values[j + target_set*nbWays], timestamp, rs, ws, tid, &isAborted); // TODO: use the whole val
						out[tid*nbTXsPerThread + i].isFound = 1;
						out[tid*nbTXsPerThread + i].val[0] = val; // TODO: value_size
						break;
					}
				}
				if(isAborted==true){ atomicAdd(&(stats->nbAbortsDataAge), 1); continue;}

				if(retry==1) start_time_commit = clock64();
				retry=TXCommit(gbc, tid, rs, ws, timestamp, wRes, retry, stats, times);
				if(retry!=oldretry) 
				{
					stop_time_commit = clock64();
					stop_time_tx = clock64();
					oldretry=retry;
				}
			}while(vote_ballot(retry) != 0);
			times[tid].runtime += stop_time_tx - start_time_tx;
			times[tid].commit  += stop_time_commit - start_time_commit;
		}
	}
}

__device__ void worker_thread(gbc_pack_t gbc_pack, SERV_ARG_DEF)
{
	uint m_warp_id = thread_id_x() / 32;

	VAR_BUF_DEF0
	uint stage_buf0 = 0;
	uint lock_id0;

	//for(stage_buf0=0; stage_buf0 < WORK_BUFF_SIZE_MAX; stage_buf0++)
	{		
		process_buffer_main_worker(gbc_pack, m_warp_id, 0, VAR_BUF0,
				stage_buf0,lock_id0, SERV_ARG);
	}
}


__global__ void server_kernel(gbc_pack_t gbc_pack, TXRecord* records, readSet* rs, writeSet* ws, warpResult* wRes, Statistics* stats, time_rate* times)
{
	__shared__ TMmetadata metadata;
	
	metadata.tp = TXRecordSize-1;
	metadata.hasWrapped = false;

	init_recv(gbc_pack);
	gc_receiver_leader(gbc_pack);
	while(1)
	{
		worker_thread(gbc_pack, &metadata, records, rs, ws, wRes, stats, times);
	}
}

__global__ void exit_kernel(gbc_t gbc)
{
		base_exit(gbc);
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
	  "  4) server config - nb threads     \n"
	  "  5) client config - nb blocks      \n"
	  "  6) client config - nb threads     \n"
	  "  7) nb TXs per thread              \n"
	  //"  8) nb repetitions                 \n"
	  "  8) prob read TX kernel            \n"
	  "  9) run duration                   \n"
	  " 10) verbose		                   \n"
	"";
	const int NB_ARGS = 11;
	int argCnt = 1;
	
	if (argc != NB_ARGS) {
		printf("%s\n", APP_HELP);
		exit(EXIT_SUCCESS);
	}

	TXRecord* records;
	writeSet* ws;
	readSet*  rs;
	warpResult* wRes;
	Statistics *d_stats, *h_stats;
	time_rate *d_times, *h_times;

	uint64_t seed = memcd_host.seed           = (long)atol(argv[argCnt++]);
	int    nbSets = memcd_host.nbSets         = (int)atol(argv[argCnt++]);
	int    nbWays = memcd_host.nbWays         = (int)atol(argv[argCnt++]);
	long   nbKeys = memcd_host.nbKeys         = memcd_host.nbSets*memcd_host.nbWays*2;
	int    nbRecvThrs 						  = (int)atol(argv[argCnt++]);
	int    nbBlks = memcd_host.nbBlocks       = (int)atol(argv[argCnt++]);
	int    nbThrs = memcd_host.nbThreads      = (int)atol(argv[argCnt++]);
	int    nbTXsT = memcd_host.nbTXsPerThread = (int)atol(argv[argCnt++]);
	//int    nbReps = memcd_host.nbReps         = (int)atol(argv[argCnt++]);
	float  prRead = memcd_host.probReadKernel = (float)atof(argv[argCnt++])/100.0;
	int    totalDuration					  = (int)atol(argv[argCnt++]);
	int    verbose 							  = (int)atol(argv[argCnt++]);

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
	gbc_pack_t gbc_pack;
	create_gbc(gbc_pack, nbBlks, nbThrs, nbRecvThrs);
	TXInit(nbThrs*nbBlks, &records, &rs, &ws, &wRes);

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

	//Start the server kernel
	cudaStream_t s2;
	cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
	server_kernel<<<1, nbRecvThrs, 0, s2>>>(gbc_pack, records, rs, ws, wRes, d_stats, d_times);
	
	// Populates the cache
	cudaStream_t s1;
	cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
	memcdWriteTx<<<nbBlks, nbThrs, 0, s1>>>(gbc_pack.gbc[CHANNEL_OFFLOAD], memcd_dev, rs, ws, wRes, d_stats, d_times);
	CUDA_CHECK_ERROR(cudaStreamSynchronize(s1), "pop sync");

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

		cudaEventRecord(start,s1);
		memcdTx<<<nbBlks, nbThrs, 0, s1>>>(gbc_pack.gbc[CHANNEL_OFFLOAD], memcd_dev, rs, ws, wRes, seed, prRead, d_stats, d_times);
		cudaEventRecord(stop,s1);

		CUDA_CHECK_ERROR(cudaEventSynchronize(stop), "in kernel");

		*memcd_host.curr_clock += 1; // increments the batch clock
		CUDA_CPY_TO_DEV(memcd_dev.curr_clock, memcd_host.curr_clock, sizeof(int));
		nbIters++;

		cudaEventElapsedTime(&tKernel_ms, start, stop);
		totT_ms += tKernel_ms;

	}
	exit_kernel<<<nbBlks, nbThrs, 0, s1>>>(gbc_pack.gbc[CHANNEL_OFFLOAD]);

	CUDA_CPY_TO_HOST(h_stats, d_stats, sizeof(Statistics));
	CUDA_CPY_TO_HOST(h_times, d_times, sizeof(time_rate)*nbThrs*nbBlks);

  	//Treat the data to generate output
	double avg_runtime=0, avg_commit=0, avg_wb=0, avg_val1=0, avg_val2=0, avg_rwb=0, avg_wait=0, avg_comp=0, avg_pv=0;
	long int totUpdates=0, totReads=0;
	for(int i=0; i<nbThrs*nbBlks; i++)
	{
		if(h_times[i].runtime < 0) printf("T%d: %li\n", i, h_times[i].runtime);
		avg_runtime += h_times[i].runtime;
		avg_commit 	+= h_times[i].commit;
		avg_wb 		+= h_times[i].dataWrite;
		avg_val1	+= h_times[i].val1;
		avg_val2	+= h_times[i].val2;
		avg_rwb		+= h_times[i].recordWrite;
		avg_wait	+= h_times[i].wait;
		avg_comp 	+= h_times[i].comparisons;
		avg_pv		+= h_times[i].preValidation;

		totUpdates 	+= h_times[i].nbUpdates;
		totReads	+= h_times[i].nbReadOnly;
	}
	avg_wait = avg_wait - avg_rwb - avg_val1 - avg_val2;

	long int denom = (long)h_stats->nbCommits*peak_clk;
	avg_runtime	/= denom;
	avg_commit 	/= denom;
	avg_wb 		/= denom;
	avg_val1 	/= denom;
	avg_val2 	/= denom;
	avg_rwb 	/= denom;
	avg_wait	/= denom;
	avg_comp	/= h_stats->nbCommits;
	avg_pv		/= denom;

	float rt_commit=0.0, rt_wb=0.0, rt_val1=0.0, rt_val2=0.0, rt_rwb=0.0, rt_wait=0.0, rt_pv=0.0;
	rt_commit	=	avg_commit / avg_runtime;
	rt_wb	 	=	avg_wb / avg_runtime;
	rt_val1	 	=	avg_val1 / avg_runtime;
	rt_val2	 	=	avg_val2 / avg_runtime;
	rt_rwb	 	=	avg_rwb / avg_runtime;
	rt_wait		=	avg_wait / avg_runtime;
	rt_pv		=	avg_pv / avg_runtime;

	int nbAborts = h_stats->nbAbortsDataAge + h_stats->nbAbortsRecordAge + h_stats->nbAbortsReadWrite + h_stats->nbAbortsPreValid;

	if(verbose)
		printf("AbortPercent\t%f %%\nThroughtput\t%f\n\nAbortDataAge\t%f %%\nAbortRecAge\t%f %%\nAbortReadWrite\t%f %%\nAbortPreVal\t%f %%\n\nTotal\t\t%f\nCommit\t\t%f\t%.2f%%\nWaitTime\t%f\t%.2f%%\nPreValidation\t%f\t%.2f%%\n1stValidation\t%f\t%.2f%%\nRecInsertVals\t%f\t%.2f%%\nRecInsert\t%f\t%.2f%%\nWriteBack\t%f\t%.2f%%\n\nComparisons\t%f\nTotalIters\t%d\n", 
			(float)nbAborts/(nbAborts+h_stats->nbCommits)*100.0,
			h_stats->nbCommits/totT_ms*1000.0,
			(float)h_stats->nbAbortsDataAge/(nbAborts+h_stats->nbCommits)*100.0,
			(float)h_stats->nbAbortsRecordAge/(nbAborts+h_stats->nbCommits)*100.0,
			(float)h_stats->nbAbortsReadWrite/(nbAborts+h_stats->nbCommits)*100.0,
			(float)h_stats->nbAbortsPreValid/(nbAborts+h_stats->nbCommits)*100.0,
			avg_runtime,
			avg_commit,
			rt_commit*100.0,
			avg_wait,
			rt_wait*100.0,
			avg_pv,
			rt_pv*100.0,
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
		printf("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%d\n", 
			(float)nbAborts/(nbAborts+h_stats->nbCommits)*100.0,
			h_stats->nbCommits/totT_ms*1000.0,
			(float)h_stats->nbAbortsDataAge/(nbAborts+h_stats->nbCommits)*100.0,
			(float)h_stats->nbAbortsRecordAge/(nbAborts+h_stats->nbCommits)*100.0,
			(float)h_stats->nbAbortsReadWrite/(nbAborts+h_stats->nbCommits)*100.0,
			(float)h_stats->nbAbortsPreValid/(nbAborts+h_stats->nbCommits)*100.0,
			avg_runtime,
			avg_commit,
			rt_commit*100.0,
			avg_wait,
			rt_wait*100.0,
			avg_pv,
			rt_pv*100.0,
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

	cudaFree(h_stats);
	cudaFree(h_times);
	cudaFree(d_stats);
	cudaFree(d_times);

	cudaFree(rs);
	cudaFree(ws);
	cudaFree(wRes);



	return EXIT_SUCCESS;
}

