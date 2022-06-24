////////////////////
////	BS		////
////////////////////

#include <time.h>
#include "API.cuh"
#include "util.cuh"
#include <unistd.h>

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

__global__ void bank_kernel(int *flag, unsigned int seed, float prRead, unsigned int roSize, unsigned int txSize, unsigned int dataSize, 
								unsigned int threadNum, VertionedDataItem* data, TXRecord* record, TMmetadata* metadata, Statistics* stats, time_rate* times)
{
	local_metadata txData;
	bool result;

	int id = threadIdx.x + blockIdx.x * blockDim.x;
	long mod = 0xFFFF;
	long rnd;
	long probRead;// = prRead * 0xFFFF;

	uint64_t state = seed+id;
	
	int value=0;
	int addr;
	//profile metrics
	long long int start_time_commit, stop_time_commit;
	long long int start_time_tx, stop_time_tx;
	long long int stop_aborted_tx=0, wastedTime=0;
	long long int start_time_total;

	long int updates=0, reads=0;
	//dijoint accesses variables
#if DISJOINT
	int min, max;
	min = dataSize/threadNum*id;
	max = dataSize/threadNum*(id+1)-1;
#endif

	while((*flag & 1)==0)
	{
		waitMem = *flag;
		wastedTime=0;
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
		do
		{	
			start_time_tx = clock64();
			TXBegin(*metadata, &txData);
			
			//Read-Only TX
			if(rnd < probRead)
			{
				value=0;
				for(int i=0; i<dataSize && txData.isAborted==false; i++)//for(int i=0; i<roSize && txData.isAborted==false; i++)//
				{
			#if DISJOINT					
					addr = RAND_R_FNC(state)%(max-min+1) + min;
			#else
					addr = RAND_R_FNC(state)%dataSize;
			#endif
					value+=TXReadOnly(data, i, &txData);
				}
				if(txData.isAborted==true)
				{
					atomicAdd(&(stats->nbAbortsDataAge), 1);
					continue;
				}
				//if(value != 10*dataSize)
				//	printf("T%d found an invariance fail: %d\n", id, value);
			}
			//Update TX
			else
			{
/*				for(int i=0; i<max(txSize,roSize) && txData.isAborted==false; i++)
				{
			#if DISJOINT					
					addr = RAND_R_FNC(state)%(max-min+1) + min;
			#else
					addr = RAND_R_FNC(state)%dataSize;
			#endif
					if(i<roSize)
						value = TXRead(data, addr, &txData);
					if(i<txSize)
						TXWrite(data, value+(1), addr, &txData);
*/
				for(int i=0; i<txSize && txData.isAborted==false; i++)
				{
			#if DISJOINT					
					addr = RAND_R_FNC(state)%(max-min+1) + min;
			#else
					addr = RAND_R_FNC(state)%dataSize;
			#endif
					value = TXRead(data, addr, &txData); 
					TXWrite(data, value-(1), addr, &txData);	

			#if DISJOINT					
					addr = RAND_R_FNC(state)%(max-min+1) + min;
			#else
					addr = RAND_R_FNC(state)%dataSize;
			#endif
					value = TXRead(data, addr, &txData); 
					TXWrite(data, value+(1), addr, &txData);
				}
				if(txData.isAborted==true)
				{
					atomicAdd(&(stats->nbAbortsDataAge), 1);
					continue;
				}
			}
			start_time_commit = clock64(); 
  			result=TXCommit(id,record,data,metadata,txData,stats,times);
  			stop_time_commit = clock64();
  			if(!result)
			{
				stop_aborted_tx = clock64();
				wastedTime += stop_aborted_tx - start_time_tx;
			}
			stop_time_tx = clock64();
		}
		while(!result);
		atomicAdd(&(stats->nbCommits), 1);
		if(txData.ws.size==0)
			reads++;
		else
			updates++;		

		times[id].total   += stop_time_tx - start_time_total;
		times[id].runtime += stop_time_tx - start_time_tx;
		times[id].commit  += stop_time_commit - start_time_commit;
		times[id].wastedTime	 += wastedTime;

	}
	times[id].nbReadOnly = reads;
	times[id].nbUpdates  = updates;
}

void getKernelOutput(Statistics *h_stats, time_rate *h_times, uint threadNum, int peak_clk, float totT_ms, uint verbose)
{
  	double avg_total=0, avg_runtime=0, avg_commit=0, avg_wb=0, avg_val1=0, avg_val2=0, avg_rwb=0, avg_comp=0, avg_waste=0;
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
		avg_waste	+= h_times[i].wastedTime;
	
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
	avg_waste	/= denom;

	float rt_commit=0.0, rt_wb=0.0, rt_val1=0.0, rt_val2=0.0, rt_rwb=0.0, dummy=0.0;
	rt_commit	=	avg_commit / avg_runtime;
	rt_wb	 	=	avg_wb / avg_runtime;
	rt_val1	 	=	avg_val1 / avg_runtime;
	rt_val2	 	=	avg_val2 / avg_runtime;
	rt_rwb	 	=	avg_rwb / avg_runtime;

	int nbAborts = h_stats->nbAbortsDataAge + h_stats->nbAbortsRecordAge + h_stats->nbAbortsReadWrite;

	
	if(verbose)
		printf("AbortPercent\t%f %%\nThroughtput\t%f\n\nAbortDataAge\t%f %%\nAbortRecAge\t%f %%\nAbortReadWrite\t%f %%\nAbortPreVal\t%f %%\n\nTotal\t\t%f\nRuntime\t\t%f\nCommit\t\t%f\t%.2f%%\nWaitTime\t%f\t%.2f%%\nPreValidation\t%f\t%.2f%%\n1stValidation\t%f\t%.2f%%\nRecInsertVals\t%f\t%.2f%%\nRecInsert\t%f\t%.2f%%\nWriteBack\t%f\t%.2f%%\nWaste\t\t%f\n\nComparisons\t%f\nTotalUpdates\t%d\nTotalReads\t%d\n", 
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
			avg_waste
			);
}

int main(int argc, char *argv[])
{
	unsigned int blockNum, threads_per_block, roSize, threadSize, dataSize, seed, verbose;
	float prRead;

	VertionedDataItem *h_data, *d_data;
	TXRecord *records;
	TMmetadata *metadata;

	Statistics *h_stats, *d_stats;
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
	threadSize			= atoi(argv[argCnt++]);
	verbose				= atoi(argv[argCnt++]);

#if DISJOINT
	dataSize=100*blockNum*threads_per_block;
#endif
	
	h_times = (time_rate*) calloc(blockNum*threads_per_block,sizeof(time_rate));
	h_stats = (Statistics*)calloc(1,sizeof(Statistics));
	h_data = (VertionedDataItem*)calloc(dataSize,sizeof(VertionedDataItem));

	//Select the GPU Device
	cudaError_t result;
	result = cudaSetDevice(0);
	if(result != cudaSuccess) fprintf(stderr, "Failed to set Device: %s\n", cudaGetErrorString(result));

	int peak_clk=1;
	cudaError_t err = cudaDeviceGetAttribute(&peak_clk, cudaDevAttrClockRate, 0);
  	if (err != cudaSuccess) {printf("cuda err: %d at line %d\n", (int)err, __LINE__); return 1;}

	
	result = TXInit(&records, &metadata);
	if(result != cudaSuccess) fprintf(stderr, "Failed TM Initialization: %s\n", cudaGetErrorString(result));
	result = cudaMalloc((void **)&d_stats, sizeof(Statistics));
	if(result != cudaSuccess) fprintf(stderr, "Failed to allocate d_stats: %s\n", cudaGetErrorString(result));
	result = cudaMalloc((void **)&d_times, blockNum*threads_per_block*sizeof(time_rate));
	if(result != cudaSuccess) fprintf(stderr, "Failed to allocate d_ratio: %s\n", cudaGetErrorString(result));
	result = cudaMalloc((void **)&d_data, dataSize*sizeof(VertionedDataItem));
	if(result != cudaSuccess) fprintf(stderr, "Failed to allocate d_data: %s\n", cudaGetErrorString(result));

	for(int i=0; i<dataSize; i++)
	{
		h_data[i].head_ptr = 1;
		h_data[i].value[h_data[i].head_ptr] = 10;
	}

	dim3 blockDist(threads_per_block,1,1);
	dim3 gridDist(blockNum, 1, 1);

	cudaMemcpy(d_data, h_data, dataSize*sizeof(VertionedDataItem), cudaMemcpyHostToDevice);
	cudaMemcpy(d_times, h_times, blockNum*threads_per_block*sizeof(time_rate), cudaMemcpyHostToDevice);
	cudaMemcpy(d_stats, h_stats, sizeof(Statistics), cudaMemcpyHostToDevice);

	float tKernel_ms = 0.0, totT_ms = 0.0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int *flag;
  	CUDA_CHECK_ERROR(cudaMallocManaged(&flag, sizeof(int)), "Could not alloc");
  	*flag = 0;

	cudaEventRecord(start); 
	bank_kernel<<<gridDist, blockDist>>>(flag, seed, prRead, roSize, threadSize, dataSize, blockNum*threads_per_block, d_data, records, metadata, d_stats, d_times);
  	cudaEventRecord(stop);
		
	//sleep for a set time to let the kernel run
	sleep(KERNEL_DURATION);
	//send termination message to the kernel and wait for a return
	__atomic_fetch_add(flag, 1, __ATOMIC_ACQ_REL);

	CUDA_CHECK_ERROR(cudaEventSynchronize(stop), "in kernel");
  	
  	cudaEventElapsedTime(&tKernel_ms, start, stop);
	totT_ms += tKernel_ms;

	//get the output performance metrics
	cudaMemcpy(h_stats, d_stats, sizeof(Statistics), cudaMemcpyDeviceToHost);
  	cudaMemcpy(h_times, d_times, blockNum*threads_per_block*sizeof(time_rate), cudaMemcpyDeviceToHost);
  	
  	getKernelOutput(h_stats, h_times, blockNum*threads_per_block, peak_clk, totT_ms, verbose);
	TXEnd(dataSize, h_data, &d_data, &records, &metadata);

	free(h_stats);
	free(h_times);
	cudaFree(d_stats);
	cudaFree(d_times);
	
	return 0;
}