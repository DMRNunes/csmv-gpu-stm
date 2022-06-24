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

__global__ void client_kernel(gbc_t gbc, uint64_t seed, uint dataSize, VertionedDataItem* data, readSet* rs, writeSet* ws, warpResult* wRes, int* ro, int roSize, int upSize, 
								Statistics* stats, time_rate* times) {

	uint tid = threadIdx.x + blockIdx.x * blockDim.x;
	uint wid = tid/32;

	init_base(_offload_tail_ptr_cpy);

	__syncthreads();

	uint64_t state = seed+tid;
	int value1, value2, timestamp, addr1, addr2, result;
	bool isAborted;

	uint dst = 0;
	uint val0 = tid;
	int val1;

	uint valid_msg, retry=1;
	uint saved_write_ptr = NOT_FOUND;

	long long int start_writeback=0, stop_writeback=0;
	long long int start_time_commit, stop_time_commit;
	long long int start_time_tx, stop_time_tx;
	long long int start_wait, stop_wait;
	long long int start_send, stop_send;

	do
	{
#if PRINT_DEBUG == 0
		if(get_lane_id()==0) printf("C%d begin\n", tid/32);
#endif
		if(retry==1)start_time_tx = clock64();
		timestamp=TXBegin(tid, ws, rs);
		isAborted=false;
		valid_msg = 1;

		//Read-Only TX
		if(ro[wid] == 1)
		{
			value1=0;
			for(int i=0; i<roSize && isAborted==false; i++)
			{
				//addr1 = RAND_R_FNC(state)%dataSize;
				value1+=TXRead(data, i, timestamp, rs, ws, tid, &isAborted);
				//printf("C%d: total so far %d\n", tid, value1);
			}
			if(isAborted==true)
			{
				atomicAdd(&(stats->nbAbortsDataAge), 1);
				continue;
			}
			//assert(value1 == 1000*dataSize);
		}
		//Update TX
		else
		{
			for(int i=0; i<upSize && isAborted==false; i++)
			{
				addr1 = RAND_R_FNC(state)%dataSize;
				value1 = TXRead(data, addr1, timestamp, rs, ws, tid, &isAborted); 
				TXWrite(data, value1-(tid*10+100), addr1, ws, tid);

				addr2 = RAND_R_FNC(state)%dataSize;
				value2 = TXRead(data, addr2, timestamp, rs, ws, tid, &isAborted); 
				TXWrite(data, value2+(tid*10+100), addr2, ws, tid);
			}
			if(isAborted==true)
			{
				atomicAdd(&(stats->nbAbortsDataAge), 1);
				continue;
			}
		}
		
		/////////////////////////////
		//Commit process
		/////////////////////////////
		if(retry==1) start_time_commit = clock64();

		if(ro[wid] == 1)
		{
			//start_send = stop_send = clock64();
			//start_wait = stop_wait = clock64();
			start_writeback = clock64();
			atomicAdd(&(stats->nbCommits), 1);
			stop_writeback = clock64();
			//stop_time_commit = clock64();
			//stop_time_tx = clock64();
			retry=0;
		}
		else
		{
			if(retry==0)
				val1=-1;
			else
				val1=timestamp;
			
			if(retry==1) start_send = clock64();
			if(get_lane_id()==0)
				saved_write_ptr = atomicAdd(&(gbc.write_ptr[dst]), 32);
			saved_write_ptr = shuffle_idx(saved_write_ptr, 0) + get_lane_id();
			do
			{
				if(base_send(gbc, dst, valid_msg, saved_write_ptr, _offload_tail_ptr_cpy, SEND_ARG))
					valid_msg = 0;
			}
			while(vote_ballot(valid_msg) != 0);
			if(retry==1) stop_send = clock64();

			if(retry==1) start_wait = clock64();
			if(get_lane_id()==0)
				while(wRes[wid].valid_entry==0);
			result = wRes[wid].lane_result[get_lane_id()];
			if(retry==1) stop_wait = clock64();


			if(result != 0)
			{
				start_writeback = clock64();
				TXWriteBack(result, data, ws[tid]);
				stop_writeback = clock64();
				stop_time_commit = clock64();
				stop_time_tx = clock64();
				atomicAdd(&(stats->nbCommits), 1);
				retry=0;
				//printf("C%d: changed %d and %d\tread %d(-%d) and %d(+%d) \n", tid, addr1, addr2, value1, tid*10+100, value2, tid*10+100);
			}
			//reset scoreboard
			if(get_lane_id()==0)
				wRes[wid].valid_entry=0;
		}
		//stop_time_commit = clock64();

#if PRINT_DEBUG == 0				
	if(thread_id_x()==0) printf("C%d: nbCommits: %d\n", blockIdx.x, stats->nbCommits);
#endif
		

	}while(vote_ballot(retry) != 0);

	times[tid].runtime 	+= stop_time_tx - start_time_tx;
	times[tid].commit 	+= stop_time_commit - start_time_commit;
	times[tid].dataWrite+= stop_writeback - start_writeback;
	times[tid].wait 	+= stop_wait - start_wait;
	times[tid].send 	+= stop_send - start_send;
	

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


__global__ void parent_kernel(uint total_sender_bk, uint sender_block_size,	uint total_recevier_bk, uint recv_block_size, gbc_pack_t gbc_pack,
								uint64_t seed, uint dataSize, VertionedDataItem* data, readSet* rs, writeSet* ws, warpResult* wRes,
								int* ro, int roSize, int upSize, Statistics* stats, time_rate* times) {

	//for(int i=0; i<SCOREBOARD_SIZE/32; i++)
	//	validSB[i]=0;
	
	cudaStream_t s2;
	cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);

	server_kernel<<<total_recevier_bk, recv_block_size, 0, s2>>>(gbc_pack, rs, ws, wRes, stats, times);

	cudaStream_t s1;
	cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);

	client_kernel<<<total_sender_bk, sender_block_size, 0, s1>>>(
			gbc_pack.gbc[CHANNEL_OFFLOAD], seed, dataSize, data, rs, ws, wRes,
			ro, roSize, upSize, stats, times);

}

void test_fine_grain_offloading(int seed, int dataSize, int client_block_size, int total_client_bk, int server_block_size, int roNum, int roSize, int upSize)
{

	int total_server_bk=1;
//	void (*server_kernel)(gbc_pack_t,
//	ALG_ARG_DEF);
//	server_kernel = server_one_phase;
	gbc_pack_t gbc_pack;
	create_gbc(gbc_pack, total_client_bk, client_block_size, server_block_size);
///////////////

	int* bankArray, *h_ro, *d_ro, aux;
	VertionedDataItem *h_data, *d_data;
	warpResult* wRes;
	readSet* rs;
	writeSet* ws;
	time_rate *h_times, *d_times;

	Statistics *h_stats, *d_stats;
	struct timespec t1,t2;
  	double elapsed_ms;

  	int peak_clk=1;
	cudaError_t err = cudaDeviceGetAttribute(&peak_clk, cudaDevAttrClockRate, 0);

	h_ro = (int*) calloc(total_client_bk*client_block_size/32, sizeof(int));
	for(int i=0; i<roNum;)
	{
		aux = rand()%(total_client_bk*client_block_size/32);
		if(h_ro[aux]==0)
		{
			h_ro[aux]=1;
			i++;
		}
	}
	h_times = (time_rate*) calloc(total_client_bk*client_block_size,sizeof(time_rate));
	h_stats = (Statistics*)calloc(1,sizeof(Statistics));


	bankArray = (int*)malloc(dataSize*sizeof(int));
	for(int i=0; i<dataSize; i++)
	{
		bankArray[i]=1000;
	}
	
	//Allocate memory in the device
	cudaError_t result;
	result = TXInit(bankArray, dataSize, client_block_size*total_client_bk, &h_data, &d_data, &rs, &ws, &wRes);
	if(result != cudaSuccess) fprintf(stderr, "Failed TM Initialization: %s\n", cudaGetErrorString(result));
	result = cudaMalloc((void **)&d_ro, total_client_bk*client_block_size/32*sizeof(int));
	if(result != cudaSuccess) fprintf(stderr, "Failed to allocate d_ro: %s\n", cudaGetErrorString(result));
	result = cudaMalloc((void **)&d_stats, sizeof(Statistics));
	if(result != cudaSuccess) fprintf(stderr, "Failed to allocate d_stats: %s\n", cudaGetErrorString(result));
	result = cudaMalloc((void **)&d_times, total_client_bk*client_block_size*sizeof(time_rate));
	if(result != cudaSuccess) fprintf(stderr, "Failed to allocate d_ratio: %s\n", cudaGetErrorString(result));

	//transfer the necessary data from the host to the device
	cudaMemcpy(d_ro, h_ro, total_client_bk*client_block_size/32*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_times, h_times, total_client_bk*client_block_size*sizeof(time_rate), cudaMemcpyHostToDevice);
	cudaMemcpy(d_stats, h_stats, sizeof(Statistics), cudaMemcpyHostToDevice);

	///////////////
	//kernel stuff
	clock_gettime(CLOCK_REALTIME, &t1);
	{
		//cudaDeviceSynchronize();

		parent_kernel<<<1, 1>>>(total_client_bk, client_block_size,
		total_server_bk, server_block_size, gbc_pack,
		1, dataSize, d_data, rs, ws, wRes,
		d_ro, roSize, upSize, d_stats, d_times);
		cudaDeviceSynchronize();
	}
	clock_gettime(CLOCK_REALTIME, &t2);
	//////////////


	free_gbc(gbc_pack);
	TXEnd(dataSize, h_data, &d_data, &rs, &ws, &wRes);

	//Copy metric data back to the host
	cudaMemcpy(h_stats, d_stats, sizeof(Statistics), cudaMemcpyDeviceToHost);
  	cudaMemcpy(h_times, d_times, total_client_bk*client_block_size*sizeof(time_rate), cudaMemcpyDeviceToHost);

  	elapsed_ms = (t2.tv_sec*1000.0 + t2.tv_nsec/1000000.0) - (t1.tv_sec*1000.0 + t1.tv_nsec/1000000.0);


	double avg_runtime=0, avg_commit=0, avg_wb=0, avg_val=0, avg_rwb=0, avg_wait=0, avg_comp=0, avg_send=0;
	for(int i=0; i<total_client_bk*client_block_size; i++)
	{
		if(h_times[i].runtime < 0) printf("T%d: %li\n", i, h_times[i].runtime);
		avg_runtime += h_times[i].runtime;
		avg_commit 	+= h_times[i].commit;
		avg_wb 		+= h_times[i].dataWrite;
		avg_val		+= h_times[i].validation;
		avg_rwb		+= h_times[i].recordWrite;
		avg_wait	+= h_times[i].wait;
		avg_comp 	+= h_times[i].comparisons;
		avg_send	+= h_times[i].send;
	}
	avg_wait = avg_wait - avg_rwb - avg_val;
	long int denom = (long)total_client_bk*client_block_size*peak_clk;
	avg_runtime	/= denom;
	avg_commit 	/= denom;
	avg_wb 		/= denom;
	avg_val 	/= denom;
	avg_rwb 	/= denom;
	avg_wait	/= denom;
	avg_comp	/= total_client_bk*client_block_size;
	avg_send	/= denom;

	float rt_commit=0.0, rt_wb=0.0, rt_val=0.0, rt_rwb=0.0, rt_wait=0.0, rt_send=0.0;
	rt_commit	=	avg_commit / avg_runtime;
	rt_wb	 	=	avg_wb / avg_runtime;
	rt_val	 	=	avg_val / avg_runtime;
	rt_rwb	 	=	avg_rwb / avg_runtime;
	rt_wait		=	avg_wait / avg_runtime;
	rt_send		=	avg_send / avg_runtime;

	int nbAborts = h_stats->nbAbortsDataAge + h_stats->nbAbortsRecordAge + h_stats->nbAbortsReadWrite + h_stats->nbAbortsWriteWrite;

	//printf("commits: %d\n", host_stats->nbCommits);

	printf("%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n", 
		(float)nbAborts/(nbAborts+h_stats->nbCommits),
		h_stats->nbCommits/elapsed_ms*1000.0,
		(float)h_stats->nbAbortsDataAge/(nbAborts+h_stats->nbCommits),
		(float)h_stats->nbAbortsRecordAge/(nbAborts+h_stats->nbCommits),
		(float)h_stats->nbAbortsReadWrite/(nbAborts+h_stats->nbCommits),
		(float)h_stats->nbAbortsWriteWrite/(nbAborts+h_stats->nbCommits),
		avg_runtime,
		avg_commit,
		rt_commit,
		avg_send,
		rt_send,
		avg_wait,
		rt_wait,
		avg_val,
		rt_val,
		avg_rwb,
		rt_rwb,
		avg_wb,
		rt_wb,
		avg_comp
		);
/*
		printf("%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n", 
		h_stats->nbCommits/elapsed_ms*1000.0,
		avg_runtime,
		avg_commit,
		avg_s1,
		avg_s2,
		avg_wait,
		//rt_commit,
		avg_val,
		//val_rate,
		avg_rwb,
		//rwb_rate,
		avg_wb
		//wb_rate,
		//wait_rate,
		//avg_comp
		);
*/
	free(h_ro);
	free(h_stats);
	free(h_times);
	cudaFree(d_ro);
	cudaFree(d_stats);
	cudaFree(d_times);
}

int main(int argc, char *argv[]) {

	int client_block_size, server_block_size;
	int total_client_bk;
	int dataSize, roNum, roSize, upSize;

	  	const char APP_HELP[] = ""                
	  "argument order:                     \n"
	  "  1) nb bank accounts               \n"
	  "  2) client config - nb threads     \n"
	  "  3) client config - nb blocks      \n"
//	  "  4) client config - TX per thread  \n"
	  "  4) server config - nb threads     \n"
	  "  5) prob read TX                   \n"
	  "  6) read TX Size                   \n"
	  "  7) update TX Size                 \n"
	"";
	const int NB_ARGS = 8;
	int argCnt = 1;
	
	if (argc != NB_ARGS) {
		printf("%s\n", APP_HELP);
		exit(EXIT_SUCCESS);
	}

	dataSize			= atoi(argv[argCnt++]);
	client_block_size	= atoi(argv[argCnt++]);
	total_client_bk	 	= atoi(argv[argCnt++]);
	server_block_size	= atoi(argv[argCnt++]);
	roNum 				= (atoi(argv[argCnt++])/100.0) * total_client_bk*client_block_size/32;
	roSize 				= atoi(argv[argCnt++]);
	upSize				= atoi(argv[argCnt++]);

	if(roNum>total_client_bk*client_block_size/32) roNum=total_client_bk*client_block_size/32;
//	printf("Number of warps: %d Read-Only Warps: %d\n", total_client_bk*client_block_size/32, roNum);

	cudaSetDevice(0);
	for (int i = 0; i < 1; i++) {
		test_fine_grain_offloading(i, dataSize, client_block_size, total_client_bk, server_block_size, roNum, roSize, upSize);
	}
	return 0;
}
