////////////////////
////	PR		////
////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define PR_MAX_RWSET_SIZE 6000

#include "pr-stm.cuh"
#include "pr-stm-internal.cuh"
#include "util.cuh"
#include <unistd.h>

typedef struct times_
{
	long long int total;
	long long int runtime;
	long long int commit;
	long int nbReadOnly;
	long int nbUpdates;	
} time_rate;

#define KERNEL_DURATION 5
#define DISJOINT 0

#define CUDA_CHECK_ERROR(func, msg) ({ \
	cudaError_t cudaError; \
	if (cudaSuccess != (cudaError = func)) { \
		fprintf(stderr, #func ": in " __FILE__ ":%i : " msg "\n   > %s\n", \
		__LINE__, cudaGetErrorString(cudaError)); \
    *((int*)0x0) = 0; /* exit(-1); */ \
	} \
  cudaError; \
})

__forceinline__ __device__ unsigned get_lane_id() {
	unsigned ret;
	asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
	return ret;
}

__device__ int waitMem;

__global__ void bank_kernel(int *flag, PR_globalKernelArgs, unsigned int seed, float prRead, unsigned int roSize, unsigned int upSize, unsigned int dataSize, 
								unsigned int threadNum, int* data, time_rate* times)
{
	//local_metadata txData;
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	long mod = 0xFFFF;
	long rnd;
	long probRead;// = prRead * 0xFFFF;

	PR_enterKernel(id);

	uint64_t state = seed+id;
	
	int value=0;
	int addr;
	//profile metrics
	long long int start_time_commit, stop_time_commit;
	long long int start_time_tx, stop_time_tx;
	long long int start_time_total;

	long int updates=0, reads=0;
	//long int totUpdates=0, totReads=0;
	//dijoint accesses variables
#if DISJOINT
	int min, max;
	min = dataSize/threadNum*id;
	max = dataSize/threadNum*(id+1)-1;
#endif

	while((*flag & 1)==0)
	{
		waitMem = *flag;
		///////
		//decide whether the thread will do update or read-only tx
		if(get_lane_id()==0)
		{
			rnd = RAND_R_FNC(state) & mod;
		}
		rnd = __shfl_sync(0xffffffff, rnd, 0);
		probRead = prRead * 0xFFFF;
		///////

		start_time_total = clock64();
		PR_txBegin();
		start_time_tx = clock64();

		//Read-Only TX
		if(rnd < probRead)
		{
			value=0;
			for(int i=0; i<dataSize; i++)		//for(int i=0; i<roSize; i++)
			{
		#if DISJOINT					
				addr = RAND_R_FNC(state)%(max-min+1) + min;
		#else
				addr = RAND_R_FNC(state)%dataSize;
		#endif
				value+=PR_read(&data[i]);
			}
			//if(value != 10*dataSize)
			//	printf("T%d found an invariance fail: %d\n", id, value);
		}
		//Update TX
		else
		{
			for(int i=0; i<upSize; i++)
			{
		#if DISJOINT					
				addr = RAND_R_FNC(state)%(max-min+1) + min;
		#else
				addr = RAND_R_FNC(state)%dataSize;
		#endif
				value = PR_read(&data[addr]);
				PR_write(&data[addr], value-1);

		#if DISJOINT					
				addr = RAND_R_FNC(state)%(max-min+1) + min;
		#else
				addr = RAND_R_FNC(state)%dataSize;
		#endif
				value = PR_read(&data[addr]);
				PR_write(&data[addr], value+1);
			}
		}
		start_time_commit = clock64(); 
		PR_txCommit();
		stop_time_commit = clock64();
		stop_time_tx = clock64();
				
		times[id].total   += stop_time_tx - start_time_total;
		times[id].runtime += stop_time_tx - start_time_tx;
		times[id].commit  += stop_time_commit - start_time_commit;

		if(rnd < probRead)
			reads++;
		else
			updates++;

	}
	times[id].nbReadOnly = reads;
	times[id].nbUpdates  = updates;

	PR_exitKernel();
}

void getKernelOutput(time_rate *h_times, uint threadNum, int peak_clk, float totT_ms, uint64_t nbCommits, uint64_t nbAborts, uint verbose)
{
  	double avg_total=0, avg_runtime=0, avg_commit=0, avg_waste=0;
  	long int totReads=0, totUpdates=0;
	
	//long int nbAborts = *PR_sumNbAborts;
	long int commits;

	for(int i=0; i<threadNum; i++)
	{
		if(h_times[i].runtime < 0) printf("T%d: %li\n", i, h_times[i].runtime);
		avg_total   += h_times[i].total;
		avg_runtime += h_times[i].runtime;
		avg_commit 	+= h_times[i].commit;
		avg_waste   += h_times[i].total - h_times[i].runtime;

		totReads 	+= h_times[i].nbReadOnly;
		totUpdates	+= h_times[i].nbUpdates;
	}
	
	nbCommits = totReads + totUpdates;
	long int denom = nbCommits*peak_clk;
	avg_total	/= denom;
	avg_runtime	/= denom;
	avg_commit 	/= denom;
	avg_waste   /= denom;

	float rt_commit=0.0;
	rt_commit	=	avg_commit / avg_runtime;

	//printf("nbCommits: %d\n", nbCommits);
	
	if(verbose)
		printf("AbortPercent\t%f %%\nThroughtput\t%f\n\nTotal\t\t%f\nRuntime\t\t%f\nCommit\t\t%f\t%.2f%%\nWaste\t\t%f\n",
			(float)nbAborts/(nbAborts+nbCommits)*100.0,
			nbCommits/totT_ms*1000.0,
			avg_total,
			avg_runtime,
			avg_commit,
			rt_commit*100.0,
			avg_waste
			);
	else
		printf("%f\t%f\t%f\t%f\t%f\t%f\t%f\n", 
			(float)nbAborts/(nbAborts+nbCommits)*100.0,
			nbCommits/totT_ms*1000.0,
			avg_total,
			avg_runtime,
			avg_commit,
			rt_commit*100.0,
			avg_waste
			);
}

void TXEnd(int dataSize, int* host_data, int** d_data)
{	
	cudaMemcpy(host_data, *d_data, dataSize*sizeof(int), cudaMemcpyDeviceToHost);
  	
  	long total=0;
  	for(int i=0; i<dataSize; i++)
  	{
  		/*
  		printf("Item: %d hp: %d tp %d\n", i, host_data[i].head_ptr, host_data[i].tail_ptr);
  		for(int j=0; j<MaxVersions; j++)
  			printf("\t%d\t%d\n", host_data[i].version[j], host_data[i].value[j]);
  		*/
  		//printf("h[%d]: %d\n", i, host_data[i]);
  		total += (long)host_data[i];
  	}
  	if(total != 10*dataSize)
  		printf("Consistency fail: Total %d\n", total);
  	//if(total != dataSize*1000)
  	//	printf("Invariance not maintanted: total bank amount is %d when it should be %d\n", total, dataSize*1000);

	free(host_data);
	cudaFree(*d_data);
}

int main(int argc, char *argv[])
{
	unsigned int blockNum, threads_per_block, roSize, upSize, dataSize, seed, verbose;
	float prRead;

	int *h_data, *d_data;
	
	//Statistics *h_stats, *d_stats;
	time_rate *d_times, *h_times;

  	const char APP_HELP[] = ""                
	  "argument order:                     \n"
	  "  1) nb bank accounts               \n"
	  "  2) client config - nb threads     \n"
	  "  3) client config - nb blocks      \n"
	  "  4) prob read TX                   \n"
	  "  5) read TX Size                   \n"
	  "  6) update TX Size                 \n"
	  "  7) verbose		                   \n"
	"";
	const int NB_ARGS = 8;
	int argCnt = 1;
	
	if (argc != NB_ARGS) {
		printf("%s\n", APP_HELP);
		exit(EXIT_SUCCESS);
	}

	seed 				= 1;
	dataSize			= atoi(argv[argCnt++]);
	threads_per_block	= atoi(argv[argCnt++]);
	blockNum		 	= atoi(argv[argCnt++]);
	prRead 				= (atoi(argv[argCnt++])/100.0);
	roSize 				= atoi(argv[argCnt++]);
	upSize				= atoi(argv[argCnt++]);
	verbose				= atoi(argv[argCnt++]);

#if DISJOINT
	dataSize=10*blockNum*threads_per_block;
#endif
	
	h_times = (time_rate*) calloc(blockNum*threads_per_block,sizeof(time_rate));
	//h_stats = (Statistics*)calloc(1,sizeof(Statistics));
	h_data = (int*)calloc(dataSize,sizeof(int));

	//Select the GPU Device
	cudaError_t result;
	result = cudaSetDevice(0);
	if(result != cudaSuccess) fprintf(stderr, "Failed to set Device: %s\n", cudaGetErrorString(result));

	int peak_clk=1;
	cudaError_t err = cudaDeviceGetAttribute(&peak_clk, cudaDevAttrClockRate, 0);
  	if (err != cudaSuccess) {printf("cuda err: %d at line %d\n", (int)err, __LINE__); return 1;}

	
	//result = TXInit(&records, &metadata);
	//if(result != cudaSuccess) fprintf(stderr, "Failed TM Initialization: %s\n", cudaGetErrorString(result));
	PR_init(1);
	pr_tx_args_s args;
	//result = cudaMalloc((void **)&d_stats, sizeof(Statistics));
	//if(result != cudaSuccess) fprintf(stderr, "Failed to allocate d_stats: %s\n", cudaGetErrorString(result));
	result = cudaMalloc((void **)&d_times, blockNum*threads_per_block*sizeof(time_rate));
	if(result != cudaSuccess) fprintf(stderr, "Failed to allocate d_ratio: %s\n", cudaGetErrorString(result));
	result = cudaMalloc((void **)&d_data, dataSize*sizeof(int));
	if(result != cudaSuccess) fprintf(stderr, "Failed to allocate d_data: %s\n", cudaGetErrorString(result));

	for(int i=0; i<dataSize; i++)
		h_data[i] = 10;

	dim3 blockDist(threads_per_block,1,1);
	dim3 gridDist(blockNum, 1, 1);

	PR_blockNum = blockNum;
	PR_threadNum = threads_per_block;

	uint64_t *sumNbAborts;
	uint64_t *sumNbCommits;

	CUDA_CHECK_ERROR(cudaMallocManaged(&sumNbCommits, sizeof(uint64_t)), "Could not alloc");
	CUDA_CHECK_ERROR(cudaMallocManaged(&sumNbAborts, sizeof(uint64_t)), "Could not alloc");

	*sumNbAborts = 0;
	*sumNbCommits = 0;

	CUDA_CPY_TO_DEV(d_data, h_data, dataSize*sizeof(int));
	CUDA_CPY_TO_DEV(d_times, h_times, blockNum*threads_per_block*sizeof(time_rate));
	//CUDA_CPY_TO_DEV(d_stats, h_stats, sizeof(Statistics));

	float tKernel_ms = 0.0, totT_ms = 0.0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int *flag;
  	CUDA_CHECK_ERROR(cudaMallocManaged(&flag, sizeof(int)), "Could not alloc");
  	*flag = 0;

	cudaEventRecord(start);
	PR_prepare_noCallback(&args);
	bank_kernel<<<gridDist, blockDist>>>(flag, args.dev, seed, prRead, roSize, upSize, dataSize, blockNum*threads_per_block, d_data, d_times);
	PR_postrun_noCallback(&args);
  	cudaEventRecord(stop);
		
	//sleep for a set time to let the kernel run
	sleep(KERNEL_DURATION);
	//send termination message to the kernel and wait for a return
	__atomic_fetch_add(flag, 1, __ATOMIC_ACQ_REL);

	CUDA_CHECK_ERROR(cudaEventSynchronize(stop), "in kernel");

	PR_reduceCommitAborts<<<PR_blockNum, PR_threadNum, 0, PR_streams[PR_currentStream]>>>
		(0, PR_currentStream, args.dev, sumNbCommits, sumNbAborts);

  	cudaEventElapsedTime(&tKernel_ms, start, stop);
	totT_ms += tKernel_ms;

	//get the output performance metrics
	//cudaMemcpy(h_stats, d_stats, sizeof(Statistics), cudaMemcpyDeviceToHost);
  	cudaMemcpy(h_times, d_times, blockNum*threads_per_block*sizeof(time_rate), cudaMemcpyDeviceToHost);
  	
//printf("aborts: %d \ncommits: %d\n", *sumNbAborts, *sumNbCommits);

  	getKernelOutput(h_times, blockNum*threads_per_block, peak_clk, totT_ms, *sumNbCommits, *sumNbAborts, verbose);
	TXEnd(dataSize, h_data, &d_data);

	//free(h_stats);
	free(h_times);
	//cudaFree(d_stats);
	cudaFree(d_times);
	
	return 0;
}