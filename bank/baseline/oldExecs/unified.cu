#include <time.h>
#include "API.cuh"
#include "util.cuh"
#include <unistd.h>

#define KERNEL_DURATION 5

#define CUDA_CHECK_ERROR(func, msg) ({ \
	cudaError_t cudaError; \
	if (cudaSuccess != (cudaError = func)) { \
		fprintf(stderr, #func ": in " __FILE__ ":%i : " msg "\n   > %s\n", \
		__LINE__, cudaGetErrorString(cudaError)); \
    *((int*)0x0) = 0; /* exit(-1); */ \
	} \
  cudaError; \
})

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

	while((*flag & 1)==0)
	{
		waitMem = *flag;
		///////
		//decide whether the warp will be do update or read-only set of txs
		rnd = RAND_R_FNC(state) & mod;
		probRead = prRead * 0xFFFF;
		///////

		do
		{	
			start_time_tx = clock64();
			TXBegin(*metadata, &txData);
//printf("t%d begins\n", id);
			if(rnd < probRead)			//Read-Only TX
			{
//printf("t%d is read only\n", id);
				for(int i=0; i<roSize && txData.isAborted==false; i++)
				{
					addr = RAND_R_FNC(state)%dataSize;
					value+=TXReadOnly(data, addr, &txData);
					//printf("t%d: ro %d addr %d\n", id, value, addr);
				}
				if(txData.isAborted==true)
				{
					atomicAdd(&(stats->nbAbortsDataAge), 1);
					continue;
				}
				//printf("t%d: ro %d\n", id, value);
			}
			else						//Update TX
			{
//printf("t%d is an update tx\n", id);
				for(int i=0; i<txSize && txData.isAborted==false; i++)
				{
					addr = RAND_R_FNC(state)%dataSize;
					value = TXRead(data, addr, &txData); //if(txData.isAborted==true) continue;
					TXWrite(data, value-(id*10+100), addr, &txData);	
//printf("t%d: took %d from %d\n", id, (id*10+100), addr);
					addr = RAND_R_FNC(state)%dataSize;
					value = TXRead(data, addr, &txData); //if(txData.isAborted==true) continue;
					TXWrite(data, value+(id*10+100), addr, &txData);
//printf("t%d: placed %d in %d\n", id, (id*10+100), addr);
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
			//atomicAdd(&(stats->nbAborts), 1);
			stop_time_tx = clock64();
		}
		while(!result);
		atomicAdd(&(stats->nbCommits), 1);
		

		times[id].runtime += stop_time_tx - start_time_tx;
		times[id].commit  += stop_time_commit - start_time_commit;

	}
}


//#include "API.cu"

int main(int argc, char *argv[])
{
	unsigned int blockNum, threads_per_block, roSize, threadSize, dataSize, seed;
	//int *h_ro, *d_ro;
	//float *h_time_rate, *d_time_rate;
	//int aux;
	float prRead;

	VertionedDataItem *h_data, *d_data;
	TXRecord *records;
	TMmetadata *metadata;

	Statistics *h_stats, *d_stats;
	time_rate *d_times, *h_times;

	//struct timespec t1,t2;
  	//double elapsed_ms;

  	const char APP_HELP[] = ""                
	  "argument order:                     \n"
	  "  1) nb bank accounts               \n"
	  "  2) client config - nb threads     \n"
	  "  3) client config - nb blocks      \n"
//	  "  4) client config - TX per thread  \n"
//	  "  4) server config - nb threads     \n"
	  "  4) prob read TX                   \n"
	  "  5) read TX Size                   \n"
	  "  6) update TX Size                 \n"
	"";
	const int NB_ARGS = 7;
	int argCnt = 1;
	
	if (argc != NB_ARGS) {
		printf("%s\n", APP_HELP);
		exit(EXIT_SUCCESS);
	}

	seed 				= 1;
	dataSize			= atoi(argv[argCnt++]);
	threads_per_block	= atoi(argv[argCnt++]);
	blockNum		 	= atoi(argv[argCnt++]);
	prRead 				= (float)atof(argv[argCnt++])/100.0;
	roSize 				= atoi(argv[argCnt++]);
	threadSize			= atoi(argv[argCnt++]);

	//if(roNum>blockNum*threads_per_block) roNum=blockNum*threads_per_block;
	//else if(roNum<0) roNum=0;

	//printf("Uniform Accesses Kernel\n");
	//printf("Seed: %d\nDataSize: %d\nThreadnb: %d\nROnb: %d\nROSize: %d\nUpdateSize: %d\n", seed, dataSize, threadNum, roNum, roSize, threadSize);

	/*h_ro = (int*) calloc(blockNum*threads_per_block, sizeof(int));
	for(int i=0; i<roNum;)
	{
		aux = rand()%(blockNum*threads_per_block);
		if(h_ro[aux]==0)
		{
			h_ro[aux]=1;
			i++;
		}
	}*/
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
	//result = cudaMalloc((void **)&d_ro, blockNum*threads_per_block*sizeof(int));
	//if(result != cudaSuccess) fprintf(stderr, "Failed to allocate d_ro: %s\n", cudaGetErrorString(result));
	result = cudaMalloc((void **)&d_stats, sizeof(Statistics));
	if(result != cudaSuccess) fprintf(stderr, "Failed to allocate d_stats: %s\n", cudaGetErrorString(result));
	result = cudaMalloc((void **)&d_times, blockNum*threads_per_block*sizeof(time_rate));
	if(result != cudaSuccess) fprintf(stderr, "Failed to allocate d_ratio: %s\n", cudaGetErrorString(result));
	result = cudaMalloc((void **)&d_data, dataSize*sizeof(VertionedDataItem));
	if(result != cudaSuccess) fprintf(stderr, "Failed to allocate d_data: %s\n", cudaGetErrorString(result));

	for(int i=0; i<dataSize; i++)
	{
		h_data[i].head_ptr = 1;
		h_data[i].value[h_data[i].head_ptr] = 1000;
	}

	dim3 blockDist(threads_per_block,1,1);
	dim3 gridDist(blockNum, 1, 1);

	cudaMemcpy(d_data, h_data, dataSize*sizeof(VertionedDataItem), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_ro, h_ro, blockNum*threads_per_block*sizeof(int), cudaMemcpyHostToDevice);
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
	TXEnd(dataSize, &h_data, &d_data, &records, &metadata);

	
	double avg_runtime=0, avg_commit=0, avg_wb=0, avg_val=0, avg_rwb=0, avg_comp=0;
	for(int i=0; i<blockNum*threads_per_block; i++)
	{
		if(h_times[i].runtime < 0) printf("T%d: %li\n", i, h_times[i].runtime);
		avg_runtime += h_times[i].runtime;
		avg_commit 	+= h_times[i].commit;
		avg_wb 		+= h_times[i].dataWrite;
		avg_val		+= h_times[i].validation;
		avg_rwb		+= h_times[i].recordWrite;
		avg_comp 	+= h_times[i].comparisons;
	}
	
	long int denom = (long)h_stats->nbCommits*peak_clk;
	avg_runtime	/= denom;
	avg_commit 	/= denom;
	avg_wb 		/= denom;
	avg_val 	/= denom;
	avg_rwb 	/= denom;
	avg_comp	/= h_stats->nbCommits;

	float rt_commit=0.0, rt_wb=0.0, rt_val=0.0, rt_rwb=0.0, dummy=0.0;
	rt_commit	=	avg_commit / avg_runtime;
	rt_wb	 	=	avg_wb / avg_runtime;
	rt_val	 	=	avg_val / avg_runtime;
	rt_rwb	 	=	avg_rwb / avg_runtime;

	int nbAborts = h_stats->nbAbortsDataAge + h_stats->nbAbortsRecordAge + h_stats->nbAbortsReadWrite;

	printf("%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n", 
		(float)nbAborts/(nbAborts+h_stats->nbCommits),
		h_stats->nbCommits/totT_ms*1000.0,
		(float)h_stats->nbAbortsDataAge/(nbAborts+h_stats->nbCommits),
		(float)h_stats->nbAbortsRecordAge/(nbAborts+h_stats->nbCommits),
		(float)h_stats->nbAbortsReadWrite/(nbAborts+h_stats->nbCommits),
		(float)h_stats->nbAbortsWriteWrite/(nbAborts+h_stats->nbCommits),
		avg_runtime,
		avg_commit,
		rt_commit,
		dummy,
		dummy,
		dummy,
		dummy,
		avg_val,
		rt_val,
		avg_rwb,
		rt_rwb,
		avg_wb,
		rt_wb,
		avg_comp
		);

	//free(h_ro);
	free(h_stats);
	free(h_times);
	//cudaFree(d_ro);
	cudaFree(d_stats);
	cudaFree(d_times);
	
	return 0;
}