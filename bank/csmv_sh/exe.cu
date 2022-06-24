////////////////////
////	SH0		////
////////////////////

#include <stdio.h>
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
#include <stdio.h>
#include <time.h>
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
#include <cuda_runtime.h>

#define SENDER_VERSION BASE_SEND

#include "../sync_lib/common.h"
#include "API.cuh"

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
#define DISJOINT 0
#define KERNEL_DURATION 5

__device__ __forceinline__ void critcal_section(SERV_ARG_DEF, uint val0, int val1)
{

	uint tid = val0;
	uint timestamp = val1;
	int result;

	//validation
	//////////////////
#if PRINT_DEBUG_ == 0
	if(get_lane_id()==0) printf("\t\t\tS%d: recv %d %d\n", thread_id_x()/32, tid/32, timestamp);
#endif

	result=TXAddToRecord(metadata, records, rs, ws, stats, times, timestamp, tid);

#if PRINT_DEBUG_ == 0
	if(get_lane_id()==0) printf("\t\t\tS%d: sent %d %d\n", thread_id_x()/32, tid/32, result);
#endif
	
	wRes[tid/32].lane_result[tid%32]=result;
	__threadfence();
	if(get_lane_id() == 0)
		wRes[tid/32].valid_entry=1;
}

#include "../sync_lib/one_phase_def.h"
#include "../sync_lib/msg_config.h"
#include "../sync_lib/msg_aux.h"
#include "../sync_lib/msg_passing.h"
#include "../sync_lib/one_phase_server.h"

__device__ int waitMem;

__global__ void client_kernel(gbc_t gbc, int *flag, uint64_t seed, uint threadNum, uint dataSize, VertionedDataItem* data, readSet* rs, writeSet* ws, warpResult* wRes, float prRead, int roSize, int upSize, 
								Statistics* stats, time_rate* times) {

	uint tid = threadIdx.x + blockIdx.x * blockDim.x;
	uint wid = tid/32;
	local_metadata txData;
	txData.prevTs=0;

	long mod = 0xFFFF;
	long rnd;
	long probRead;

	init_base(_offload_tail_ptr_cpy);

	__syncthreads();

	uint64_t state = seed+tid;
	int value1, value2, addr1, addr2, result;

	uint dst = 0;
	uint val0 = tid;
	int val1;
	uint retry=1;

	long long int start_writeback=0, stop_writeback=0;
	long long int start_time_commit, stop_time_commit;
	long long int start_time_tx, start_time_total, stop_time_tx;
	long long int start_wait=0, stop_wait=0;
	long long int stop_aborted_tx=0, wastedTime=0;

	long int updates=0, reads=0;
	//disjoint accesses variables
#if DISJOINT
	int min, max;
	min = dataSize/threadNum*tid;
	max = dataSize/threadNum*(tid+1)-1;
#endif

	while((*flag & 1)==0)
	{
		waitMem = *flag;
		retry=1;
		wastedTime=0;
		///////
		//decide whether the warp will be do update or read-only set of txs
		if(get_lane_id()==0)
		{
			rnd = RAND_R_FNC(state) & mod;
		}
		rnd = __shfl_sync(0xffffffff, rnd, 0);
		probRead = prRead * 0xFFFF;
		///////
		start_time_total = clock64();
		do
		{
			if(retry==1)start_time_tx = clock64();
			TXBegin(tid, ws, rs, &txData);

			//Read-Only TX
			if(rnd < probRead)
			{
				value1=0;
				for(int i=0; i<dataSize && txData.isAborted==false; i++)//for(int i=0; i<roSize && txData.isAborted==false; i++)
				{
			#if DISJOINT					
					addr1 = RAND_R_FNC(state)%(max-min+1) + min;
			#else
					addr1 = RAND_R_FNC(state)%dataSize;
			#endif
					value1+=TXReadOnly(data, i, &txData, rs, tid);
				}
				if(txData.isAborted==true)
				{
					atomicAdd(&(stats->nbAbortsDataAge), 1);
					continue;
				}
				//if(value1 != 100*dataSize)
				//	printf("T%d found an invariance fail: %d\n", tid, value1);
				//assert(value1 == 1000*dataSize);
			}
			//Update TX
			else
			{
/*				for(int i=0; i<max(upSize,roSize) && txData.isAborted==false; i++)
				{
			#if DISJOINT					
					addr1 = RAND_R_FNC(state)%(max-min+1) + min;
			#else
					addr1 = RAND_R_FNC(state)%dataSize;
			#endif
					if(i<roSize)
						value1 = TXRead(data, addr1, &txData, rs, ws, tid);
					if(i<upSize)
						TXWrite(data, value1+(1), addr1, ws, tid);
*/
				for(int i=0; i<upSize && txData.isAborted==false; i++)
				{
			#if DISJOINT					
					addr1 = RAND_R_FNC(state)%(max-min+1) + min;
			#else
					addr1 = RAND_R_FNC(state)%dataSize;
			#endif
					value1 = TXRead(data, addr1, &txData, rs, ws, tid); 
					TXWrite(data, value1-(1), addr1, ws, tid);

			#if DISJOINT					
					addr2 = RAND_R_FNC(state)%(max-min+1) + min;
			#else
					addr2 = RAND_R_FNC(state)%dataSize;
			#endif
					value2 = TXRead(data, addr2, &txData, rs, ws, tid); 
					TXWrite(data, value2+(1), addr2, ws, tid);
				}
				if(txData.isAborted==true)
				{
					atomicAdd(&(stats->nbAbortsDataAge), 1);
					continue;
				}
			}
			
			/////////////////////////////
			//Commit process
			/////////////////////////////
			if(retry==1) start_time_commit = clock64();
			if(rnd < probRead)
			{
				start_wait = stop_wait = clock64();
				start_writeback = clock64();
				atomicAdd(&(stats->nbCommits), 1);
				stop_writeback = clock64();
				stop_time_commit = clock64();
				stop_time_tx = clock64();
				retry=0;
				reads++;
			}
			else
			{
				if(retry==0)
					val1=-1;
				else
					val1=txData.timestamp;
				
				if(retry==1) start_wait = clock64();
				if(get_lane_id()==0)
					txData.saved_write_ptr = atomicAdd(&(gbc.write_ptr[dst]), 32);
				txData.saved_write_ptr = shuffle_idx(txData.saved_write_ptr, 0) + get_lane_id();
				do
				{
					if(base_send(gbc, dst, txData.valid_msg, txData.saved_write_ptr, _offload_tail_ptr_cpy, SEND_ARG))
						txData.valid_msg = 0;
				}
				while(vote_ballot(txData.valid_msg) != 0);
				
				if(get_lane_id()==0)
					while(wRes[wid].valid_entry==0);
				result = wRes[wid].lane_result[get_lane_id()];
				if(retry==1) stop_wait = clock64();

				if(result != 0)
				{
					start_writeback 	= clock64();
					TXWriteBack(result, data, ws[tid]);
					stop_writeback 		= clock64();
					stop_time_commit 	= clock64();
					stop_time_tx 		= clock64();
					atomicAdd(&(stats->nbCommits), 1);
					
					txData.prevTs = result;
					retry=0;
					updates++;
				}
				else if(retry==1)
				{
					stop_aborted_tx = clock64();
					wastedTime += stop_aborted_tx - start_time_tx;
				}
				//reset scoreboard
				if(get_lane_id()==0)
					wRes[wid].valid_entry=0;
			}
		}while(vote_ballot(retry) != 0);

		times[tid].total   		 += stop_time_tx - start_time_total;
		times[tid].runtime 	+= stop_time_tx - start_time_tx;
		times[tid].commit 	+= stop_time_commit - start_time_commit;
		times[tid].dataWrite+= stop_writeback - start_writeback;
		times[tid].wait 	+= stop_wait - start_wait;
		times[tid].wastedTime	 += wastedTime;
	}

	times[tid].nbReadOnly = reads;
	times[tid].nbUpdates  = updates;

	//exit process
	base_exit(gbc);
}

__device__ void worker_thread(gbc_pack_t gbc_pack, SERV_ARG_DEF, uint stage_buf0)
{
	uint m_warp_id = thread_id_x() / 32;

	VAR_BUF_DEF0
	//uint stage_buf0 = 0;
	uint lock_id0;

	//for(stage_buf0=0; stage_buf0 < WORK_BUFF_SIZE_MAX; stage_buf0++)
	{		
		process_buffer_main_worker(gbc_pack, m_warp_id, 0, VAR_BUF0,
				stage_buf0,lock_id0, SERV_ARG);
	}
}


__global__ void server_kernel(gbc_pack_t gbc_pack, readSet* rs, writeSet* ws, warpResult* wRes, Statistics* stats, time_rate* times)
{
	__shared__ TMmetadata metadata;
	__shared__ TXRecord records[TXRecordSize];

	init_recv(gbc_pack);
	uint stage_buf=0;

	gc_receiver_leader(gbc_pack);
	while(1)
	{
		worker_thread(gbc_pack, &metadata, records, rs, ws, wRes, stats, times, stage_buf);
		//if(++stage_buf>=16) stage_buf=0;
	}
}


__global__ void parent_kernel(int *flag, uint total_sender_bk, uint sender_block_size,	uint total_recevier_bk, uint recv_block_size, gbc_pack_t gbc_pack,
								uint64_t seed, uint dataSize, VertionedDataItem* data, readSet* rs, writeSet* ws, warpResult* wRes,
								float prRead, int roSize, int upSize, Statistics* stats, time_rate* times) {

	//for(int i=0; i<SCOREBOARD_SIZE/32; i++)
	//	validSB[i]=0;
	
	cudaStream_t s2;
	cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);

	server_kernel<<<total_recevier_bk, recv_block_size, 0, s2>>>(gbc_pack, rs, ws, wRes, stats, times);

	cudaStream_t s1;
	cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);

	client_kernel<<<total_sender_bk, sender_block_size, 0, s1>>>(
			gbc_pack.gbc[CHANNEL_OFFLOAD], flag, seed, total_sender_bk*sender_block_size, dataSize, data, rs, ws, wRes,
			prRead, roSize, upSize, stats, times);

}

void test_fine_grain_offloading(int seed, int dataSize, int client_block_size, int total_client_bk, int server_block_size, float prRead, int roSize, int upSize, int verbose)
{

	int total_server_bk=1;
//	void (*server_kernel)(gbc_pack_t,
//	ALG_ARG_DEF);
//	server_kernel = server_one_phase;
	gbc_pack_t gbc_pack;
	create_gbc(gbc_pack, total_client_bk, client_block_size, server_block_size);
///////////////

	int* bankArray;
	VertionedDataItem *h_data, *d_data;
	warpResult* wRes;
	readSet* rs;
	writeSet* ws;

	time_rate *h_times, *d_times;
	Statistics *h_stats, *d_stats;

  	//time measurement variables
  	int peak_clk=1;
	cudaError_t err = cudaDeviceGetAttribute(&peak_clk, cudaDevAttrClockRate, 0);
	float tKernel_ms = 0.0, totT_ms = 0.0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	h_times = (time_rate*) calloc(total_client_bk*client_block_size,sizeof(time_rate));
	h_stats = (Statistics*)calloc(1,sizeof(Statistics));


	bankArray = (int*)malloc(dataSize*sizeof(int));
	for(int i=0; i<dataSize; i++)
	{
		bankArray[i]=100;
	}
	
	//Allocate memory in the device
	cudaError_t result;
	result = TXInit(bankArray, dataSize, client_block_size*total_client_bk, &h_data, &d_data, &rs, &ws, &wRes);
	if(result != cudaSuccess) fprintf(stderr, "Failed TM Initialization: %s\n", cudaGetErrorString(result));
	result = cudaMalloc((void **)&d_stats, sizeof(Statistics));
	if(result != cudaSuccess) fprintf(stderr, "Failed to allocate d_stats: %s\n", cudaGetErrorString(result));
	result = cudaMalloc((void **)&d_times, total_client_bk*client_block_size*sizeof(time_rate));
	if(result != cudaSuccess) fprintf(stderr, "Failed to allocate d_ratio: %s\n", cudaGetErrorString(result));

	//transfer the necessary data from the host to the device
	cudaMemcpy(d_times, h_times, total_client_bk*client_block_size*sizeof(time_rate), cudaMemcpyHostToDevice);
	cudaMemcpy(d_stats, h_stats, sizeof(Statistics), cudaMemcpyHostToDevice);

	int *flag;
  	CUDA_CHECK_ERROR(cudaMallocManaged(&flag, sizeof(int)), "Could not alloc");
  	*flag = 0;

	///////////////
	//kernel stuff
  	cudaEventRecord(start);
	parent_kernel<<<1, 1>>>(flag, total_client_bk, client_block_size,
								total_server_bk, server_block_size, gbc_pack,
								1, dataSize, d_data, rs, ws, wRes,
								prRead, roSize, upSize, d_stats, d_times);
	cudaEventRecord(stop);
	//sleep for a set time to let the kernel run
	sleep(KERNEL_DURATION);
	//send termination message to the kernel and wait for a return
	__atomic_fetch_add(flag, 1, __ATOMIC_ACQ_REL);

	CUDA_CHECK_ERROR(cudaEventSynchronize(stop), "in kernel");
	//////////////
	
	//Take the time the kernel took to complete
	cudaEventElapsedTime(&tKernel_ms, start, stop);
	totT_ms += tKernel_ms;


	free_gbc(gbc_pack);
	TXEnd(dataSize, h_data, &d_data, &rs, &ws, &wRes);

	//Copy metric data back to the host
	cudaMemcpy(h_stats, d_stats, sizeof(Statistics), cudaMemcpyDeviceToHost);
  	cudaMemcpy(h_times, d_times, total_client_bk*client_block_size*sizeof(time_rate), cudaMemcpyDeviceToHost);

  	//Treat the data to generate output
	double avg_total=0, avg_runtime=0, avg_commit=0, avg_wb=0, avg_val1=0, avg_val2=0, avg_rwb=0, avg_wait=0, avg_comp=0, avg_pv=0, avg_waste=0;
	long int totUpdates=0, totReads=0;
	for(int i=0; i<total_client_bk*client_block_size; i++)
	{
		if(h_times[i].runtime < 0) printf("T%d: %li\n", i, h_times[i].runtime);
		avg_total   += h_times[i].total;
		avg_runtime += h_times[i].runtime;
		avg_commit 	+= h_times[i].commit;
		avg_wb 		+= h_times[i].dataWrite;
		avg_val1	+= h_times[i].val1;
		avg_val2	+= h_times[i].val2;
		avg_rwb		+= h_times[i].recordWrite;
		avg_wait	+= h_times[i].wait;
		avg_comp 	+= h_times[i].comparisons;
		avg_waste	+= h_times[i].wastedTime;

		totUpdates 	+= h_times[i].nbUpdates;
		totReads	+= h_times[i].nbReadOnly;
	}
	avg_wait = avg_wait - avg_rwb - avg_val1 - avg_val2;

	long int denom = (long)h_stats->nbCommits*peak_clk;
	avg_total   /= denom;
	avg_runtime	/= denom;
	avg_commit 	/= denom;
	avg_wb 		/= denom;
	avg_val1 	/= denom;
	avg_val2 	/= denom;
	avg_rwb 	/= denom;
	avg_wait	/= denom;
	avg_comp	/= h_stats->nbCommits;
	avg_pv		/= denom;
	avg_waste	/= denom;

	float rt_commit=0.0, rt_wb=0.0, rt_val1=0.0, rt_val2=0.0, rt_rwb=0.0, rt_wait=0.0, rt_pv=0.0;
	rt_commit	=	avg_commit / avg_runtime;
	rt_wb	 	=	avg_wb / avg_runtime;
	rt_val1	 	=	avg_val1 / avg_runtime;
	rt_val2	 	=	avg_val2 / avg_runtime;
	rt_rwb	 	=	avg_rwb / avg_runtime;
	rt_wait		=	avg_wait / avg_runtime;
	rt_pv		=	avg_pv / avg_runtime;

	int nbAborts = h_stats->nbAbortsDataAge + h_stats->nbAbortsRecordAge + h_stats->nbAbortsReadWrite + h_stats->nbAbortsWriteWrite;

	if(verbose)
		printf("AbortPercent\t%f %%\nThroughtput\t%f\n\nAbortDataAge\t%f %%\nAbortRecAge\t%f %%\nAbortReadWrite\t%f%%\nAbortPreVal\t%f %%\n\nTotal\t\t%f\nRuntime\t\t%f\nCommit\t\t%f\t%.2f%%\nWaitTime\t%f\t%.2f%%\nPreValidation\t%f\t%.2f%%\n1stValidation\t%f\t%.2f%%\nRecInsertVals\t%f\t%.2f%%\nRecInsert\t%f\t%.2f%%\nWriteBack\t%f\t%.2f%%\nWaste\t\t%f\n\nComparisons\t%f\nTotalUpdates\t%d\nTotalReads\t%d\n", 
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
			avg_waste,
			avg_comp,
			totUpdates,
			totReads
			);
	else
		printf("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", 
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
			avg_waste
			);

	free(h_stats);
	free(h_times);
	cudaFree(d_stats);
	cudaFree(d_times);
}

int main(int argc, char *argv[]) {

	int client_block_size, server_block_size, verbose;
	int total_client_bk;
	int dataSize, roSize, upSize;
	float prRead;

	  	const char APP_HELP[] = ""                
	  "argument order:                     \n"
	  "  1) nb bank accounts               \n"
	  "  2) client config - nb threads     \n"
	  "  3) client config - nb blocks      \n"
	  "  4) server config - nb threads     \n"
	  "  5) prob read TX                   \n"
	  "  6) read TX Size                   \n"
	  "  7) update TX Size                 \n"
	  "  8) verbose		                   \n"
	"";
	const int NB_ARGS = 9;
	int argCnt = 1;
	
	if (argc != NB_ARGS) {
		printf("%s\n", APP_HELP);
		exit(EXIT_SUCCESS);
	}

	dataSize			= atoi(argv[argCnt++]);
	client_block_size	= atoi(argv[argCnt++]);
	total_client_bk	 	= atoi(argv[argCnt++]);
	server_block_size	= atoi(argv[argCnt++]);
	prRead 				= (atoi(argv[argCnt++])/100.0);
	roSize 				= atoi(argv[argCnt++]);
	upSize				= atoi(argv[argCnt++]);
	verbose				= atoi(argv[argCnt++]);

#if DISJOINT
	dataSize=10*total_client_bk*client_block_size;
#endif

	cudaSetDevice(0);
	for (int i = 0; i < 1; i++) {
		test_fine_grain_offloading(i, dataSize, client_block_size, total_client_bk, server_block_size, prRead, roSize, upSize, verbose);
	}
	return 0;
}
