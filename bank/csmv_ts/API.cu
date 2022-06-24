#include "API.cuh"
__device__ volatile uint globalClock=0;
__device__ volatile uint attemptGC=0;

__device__ int TXRead(VertionedDataItem* data, int addr, uint timestamp, readSet* rs, writeSet* ws, uint tid, bool* isAborted)
{
	int value_read;

	for(int i=0; i<ws[tid].size; i++)
	{
		if(addr == ws[tid].addrs[i])
		{
			value_read = ws[tid].value[i];
			rs[tid].addrs[rs[tid].size++]=addr;

			return value_read;
		}
	}

	if(timestamp < data[addr].version[data[addr].tail_ptr])
	{
		*isAborted=true;
		return 0;
	}

	for(int i=data[addr].head_ptr;; i=bufDec(i))
	{
		if(data[addr].version[i] <= timestamp)
		{
				value_read = data[addr].value[i];
				rs[tid].addrs[rs[tid].size++] = addr;
				return value_read;
		}

		if(i == data[addr].tail_ptr)
			break;
	}

	*isAborted=true;
	return 0;
}

__device__ int TXReadOnly(VertionedDataItem* data, int addr, uint timestamp, readSet* rs, writeSet* ws, uint tid, bool* isAborted)
{
	int value_read;

	if(timestamp < data[addr].version[data[addr].tail_ptr])
	{
		*isAborted=true;
		return 0;
	}

	for(int i=data[addr].head_ptr;; i=bufDec(i))
	{
		if(data[addr].version[i] <= timestamp)
		{
				value_read = data[addr].value[i];
				return value_read;
		}

		if(i == data[addr].tail_ptr)
			break;
	}

	*isAborted=true;
	return 0;
}

__device__ bool TXWrite(VertionedDataItem* data, int value, int addr, writeSet* ws, uint tid)
{
	for(int i=0; i < ws[tid].size; i++)
	{
		if(addr == ws[tid].addrs[i])
		{
			ws[tid].value[i] = value;
			return true;
		}
	}

	ws[tid].addrs[ws[tid].size] = addr;
	ws[tid].value[ws[tid].size] = value;
	ws[tid].size++;

	return true;
}


__device__ bool TXValidate(int timestamp, uint* txNumber, TXRecord* TxRecords, readSet read_log, writeSet write_log, int hp, int tp, int oldhp, Statistics* stats, int tid, time_rate* times)
{
	long int comparisons=0;
	int aux;

	while((aux=txNumber[tp])>txNumber[hp])
		tp=advance_pointer(tp);
	 
	if(aux > timestamp)
	{
		atomicAdd(&(stats->nbAbortsRecordAge), 1);
		return false;
	}
	

	//validate read set
	for (int i = hp; i != oldhp; i=decrease_pointer(i))
	{
		if(txNumber[i] <= timestamp)
			break;

		comparisons++;
		for(int j=0; j < TxRecords[i].n_writes; j++)
		{
			for (int k = 0; k < read_log.size; k++)
			{
				if(read_log.addrs[k] == TxRecords[i].writeSet[j])
				{
					atomicAdd(&(stats->nbAbortsReadWrite), 1);
					times[tid].comparisons += comparisons;
					return false;
				}
			}
		}
	}
	times[tid].comparisons += comparisons;

	//check for consecutive writes to the same position by transactions that are still committing
/*	for (int i = hp; i != oldhp; i=decrease_pointer(i))
	{
		if((TxRecords[i].recordCommitted) || (TxRecords[tp].transactionNumber <= timestamp))
			break;

		for(int j=0; j < TxRecords[i].n_writes; j++)
		{
			for (int k = 0; k < write_log.size; k++)
			{
				if(write_log.addrs[k] == TxRecords[i].writeSet[j])
				{
					atomicAdd(&(stats->nbAbortsWriteWrite), 1);
					return false;
				}
			}
		}
	}
*/
	return true;	
}


__device__ void TXWriteBack(int newtimestamp, VertionedDataItem* data, writeSet write_log)
{
	uint readAGC;
	uint readGC;

	do{
		readAGC = attemptGC;
		if(newtimestamp > readAGC)
			//atomicCAS((uint*)&attemptGC, readAGC, newtimestamp);
			attemptGC = newtimestamp;
	}
	while(newtimestamp > attemptGC);

	for (int i = 0; i < write_log.size; i++)
	{
		if(bufInc(data[write_log.addrs[i]].head_ptr) == data[write_log.addrs[i]].tail_ptr)
			data[write_log.addrs[i]].tail_ptr=bufInc(data[write_log.addrs[i]].tail_ptr);
	}
	__threadfence();

	for (int i = 0; i < write_log.size; i++)
	{
		data[write_log.addrs[i]].version[ bufInc(data[write_log.addrs[i]].head_ptr) ] = newtimestamp;
		data[write_log.addrs[i]].value  [ bufInc(data[write_log.addrs[i]].head_ptr) ] = write_log.value[i];
	}
	__threadfence();

	for (int i = 0; i < write_log.size; i++)
	{
		data[write_log.addrs[i]].head_ptr=bufInc(data[write_log.addrs[i]].head_ptr);
	}
	__threadfence();
	
	readGC = globalClock;
	if(attemptGC <= newtimestamp)
		do{
			atomicCAS((uint*)&globalClock, readGC, newtimestamp);
			readGC = globalClock;
		}while(readGC < newtimestamp);
}

//returns commit timestamp
__device__ int TXAddToRecord(TMmetadata* metadata, uint* txNumber, TXRecord* TxRecords, readSet* read_log, writeSet* write_log, Statistics* stats, time_rate* times, int timestamp, int tid)
{

	// with a bounded data structure that can eject TxRecords unpredictably (due to lack of space) the first thing to check
	// is if the  TX being validated has a smaller timestamp than the oldest timestamp in the TxRecords => if so we have to abort to be on the safe side
	long long int start_time_recWrite, stop_time_recWrite;
	long long int start_time_validation, stop_time_validation;
	//long long int start_time, stop_time;

	int curr_r_hp = metadata->r_hp;
	int curr_w_hp = metadata->w_hp;
	int old_r_hp = curr_r_hp;
	int newtimestamp;

	if(timestamp==-1)
		return 0;
	
	start_time_validation = clock64();
	if(!TXValidate(timestamp, txNumber, TxRecords, read_log[tid], write_log[tid], curr_r_hp, metadata->tp, metadata->tp, stats, tid, times))
		return 0;
	stop_time_validation = clock64();
//	printf("t%d: validation successful\n", id);

	start_time_recWrite = clock64();
	bool isSet=false;
	do
	{
		curr_r_hp = metadata->r_hp;
		curr_w_hp = metadata->w_hp;

		if(curr_r_hp != old_r_hp)
		{
			if(TXValidate(timestamp, txNumber, TxRecords, read_log[tid], write_log[tid], curr_r_hp, metadata->tp, old_r_hp, stats, tid, times)==false)
			{	
				return 0;
			}
			old_r_hp = curr_r_hp;
		}
		if(curr_r_hp != curr_w_hp)
		{
			continue;
		}
		if(isSet=(atomicCAS(&(metadata->w_hp), curr_w_hp, advance_pointer(curr_w_hp)) == curr_w_hp))
		{	
			
			if(++curr_w_hp == TXRecordSize)
				curr_w_hp = 0;

			//if the transaction to be replaced has not yet finish writting to the global memory, spin until it does
			//if(TxRecords[curr_w_hp].transactionNumber > 0)
			//{
			//	while(TxRecords[curr_w_hp].recordCommitted==false);	
			//}	
			
			//advance the tail pointer before replacing the values, so that validating transactions do not access transitory data
			if(curr_w_hp == metadata->tp)
				metadata->tp = advance_pointer(metadata->tp);
			__threadfence();

			//write the WS in position curr_w_hp
			newtimestamp=txNumber[decrease_pointer(curr_w_hp)]+1;
			txNumber[curr_w_hp] = newtimestamp;
			TxRecords[curr_w_hp].n_writes = write_log[tid].size;
			//TxRecords[curr_w_hp].recordCommitted = false;
			//printf("T%d/%d inserting in the record: in pos %d val %d\n", tid, threadIdx.x, curr_w_hp, newtimestamp);
			for(int i=0; i < write_log[tid].size; i++)
			{
				TxRecords[curr_w_hp].writeSet[i] = write_log[tid].addrs[i];
			}
			__threadfence();
			metadata->r_hp = advance_pointer(metadata->r_hp);
			stop_time_recWrite=clock64();
		}
	}while(!isSet);

	times[tid].validation  += stop_time_validation - start_time_validation;
	times[tid].recordWrite += stop_time_recWrite - start_time_recWrite;

	return newtimestamp;			
}


__device__ uint TXBegin(uint tid, writeSet* ws, readSet* rs)
{
	rs[tid].size=0;
	ws[tid].size=0;
	return globalClock;
}

cudaError_t TXInit(int* dataArray, uint dataSize, uint threadNum, VertionedDataItem** host_data, VertionedDataItem** d_data, readSet** d_rs, writeSet** d_ws, TXRecord** d_records, warpResult** d_wRes)
{
	//GlobalData* dev_data;
	VertionedDataItem*	h_data;
	TXRecord*			h_records;
	writeSet*			h_ws;
	readSet*			h_rs;
	warpResult*			h_wRes;

	cudaError_t result;
	result = cudaMalloc((void **)d_data, dataSize*sizeof(VertionedDataItem));
	if(result != cudaSuccess) fprintf(stderr, "Failed to allocate d_data: %s\n", cudaGetErrorString(result));
	result = cudaMalloc((void **)d_rs, threadNum*sizeof(readSet));
	if(result != cudaSuccess) fprintf(stderr, "Failed to allocate readSet: %s\n", cudaGetErrorString(result));
	result = cudaMalloc((void **)d_ws, threadNum*sizeof(writeSet));
	if(result != cudaSuccess) fprintf(stderr, "Failed to allocate d_metad: %s\n", cudaGetErrorString(result));
	result = cudaMalloc((void **)d_records, TXRecordSize*sizeof(TXRecord));
	if(result != cudaSuccess) fprintf(stderr, "Failed to allocate d_records: %s\n", cudaGetErrorString(result));
	result = cudaMalloc((void **)d_wRes, threadNum/32*sizeof(warpResult));
	if(result != cudaSuccess) fprintf(stderr, "Failed to allocate wRes: %s\n", cudaGetErrorString(result));

	h_data = (VertionedDataItem*)calloc(dataSize,sizeof(VertionedDataItem));
	for(int i=0; i<dataSize; i++)
	{
		h_data[i].head_ptr = 1;
		h_data[i].value[h_data[i].head_ptr] = dataArray[i];
	}
	
	*host_data = h_data;
	h_records  = (TXRecord*) calloc(TXRecordSize, sizeof(TXRecord));
	h_ws       = (writeSet*) calloc(threadNum, sizeof(writeSet));
	h_rs       = (readSet*)  calloc(threadNum, sizeof(readSet));
	h_wRes     = (warpResult*)   calloc(threadNum/32, sizeof(warpResult));

	cudaMemcpy(*d_data, h_data, dataSize*sizeof(VertionedDataItem), cudaMemcpyHostToDevice);
	cudaMemcpy(*d_records, h_records, TXRecordSize*sizeof(TXRecord), cudaMemcpyHostToDevice);
	cudaMemcpy(*d_ws, h_ws, threadNum*sizeof(writeSet), cudaMemcpyHostToDevice);
	cudaMemcpy(*d_rs, h_rs, threadNum*sizeof(readSet), cudaMemcpyHostToDevice);
	cudaMemcpy(*d_wRes, h_wRes, threadNum/32*sizeof(warpResult), cudaMemcpyHostToDevice);

	free(h_records);
	free(h_ws);
	free(h_rs);
	free(h_wRes);

	return result;
}

void TXEnd(int dataSize, VertionedDataItem* host_data, VertionedDataItem** d_data, readSet** d_rs, writeSet** d_ws, warpResult** d_wRes)
{	
  	cudaMemcpy(host_data, *d_data, dataSize*sizeof(VertionedDataItem), cudaMemcpyDeviceToHost);

  	long total=0;
  	for(int i=0; i<dataSize; i++)
  	{
  		/*
  		printf("Item: %d\n", i);
  		for(int j=0; j<MaxVersions; j++)
  		{
  			printf("\t%d\t%d\t", host_data[i].version[j], host_data[i].value[j]);
  			if(j==host_data[i].head_ptr) printf("H");
  			if(j==host_data[i].tail_ptr) printf("T");
  			printf("\n");
  		}
  		*/
  		total += (long)host_data[i].value[host_data[i].head_ptr];
  	}
  	if(total != dataSize*1000)
  		printf("Invariance not maintanted: total bank amount is %d when it should be %d\n", total, dataSize*1000);

	free(host_data);
	cudaFree(*d_data);
	cudaFree(*d_rs);
	cudaFree(*d_ws);
	cudaFree(*d_wRes);
}



/*
cudaError_t TXInit(curandState *state, int* roArr, int roSize, int txSize, int threadNum, int* dataArray, int dataSize, VertionedDataItem* dev_data, TXRecord* tx_records, TMmetadata* metadata)
{
	//GlobalData* dev_data;
	VertionedDataItem* host_data;
	TXRecord* 	host_records;
	TMmetadata*  host_metadata;
	int *d_roArr;

	cudaError_t result;
	result = cudaMalloc((void **)&dev_data, dataSize*sizeof(VertionedDataItem));
	if(result != cudaSuccess) fprintf(stderr, "Failed to allocate dev_data: %s\n", cudaGetErrorString(result));

	result = cudaMalloc((void **)&tx_records,  TXRecordSize*sizeof(TXRecord));
	if(result != cudaSuccess) fprintf(stderr, "Failed to allocate tx_records: %s\n", cudaGetErrorString(result));

	result = cudaMalloc((void **)&metadata, sizeof(TMmetadata));
	if(result != cudaSuccess) fprintf(stderr, "Failed to allocate metadata: %s\n", cudaGetErrorString(result));

	result = cudaMalloc((void **)&d_roArr, threadNum*sizeof(int));
	if(result != cudaSuccess) fprintf(stderr, "Failed to allocate roArr: %s\n", cudaGetErrorString(result));

	printf("\nStarting vector\n");
	host_data = (VertionedDataItem*)calloc(dataSize,sizeof(VertionedDataItem));
	for(int i=0; i<dataSize; i++)
	{
		host_data[i].head_ptr = 1;
		host_data[i].tail_ptr = 0;
		host_data[i].value[host_data[i].head_ptr] = dataArray[i];
		host_data[i].version[host_data[i].head_ptr] = 0;
		printf("Addr %d value: %d version: %d\n", i, host_data[i].value[host_data[i].head_ptr], host_data[i].version[host_data[i].head_ptr]);
	}
	
	host_records  = (TXRecord*)   calloc(TXRecordSize, sizeof(TXRecord));
	host_metadata = (TMmetadata*) calloc(1, sizeof(TMmetadata));

	cudaMemcpy(dev_data, host_data, dataSize*sizeof(VertionedDataItem), cudaMemcpyHostToDevice);
	cudaMemcpy(tx_records, host_records, TXRecordSize*sizeof(TXRecord), cudaMemcpyHostToDevice);
	cudaMemcpy(metadata, host_metadata, sizeof(TMmetadata), cudaMemcpyHostToDevice);
	cudaMemcpy(d_roArr, roArr, threadNum*sizeof(int), cudaMemcpyHostToDevice);

	counter_kernel<<<(threadNum+threads_per_block-1)/threads_per_block, threads_per_block>>>(state, d_roArr, roSize, txSize, dataSize, threadNum, dev_data, tx_records, metadata);
  	cudaDeviceSynchronize();
	
  	cudaMemcpy(host_data, dev_data, 1*sizeof(VertionedDataItem), cudaMemcpyDeviceToHost);

  	printf("\nFinal value\n");
	for(int i=0; i<dataSize; i++)
	{
	  	for(int j=0; j<MaxVersions; j++)
	  	{
	  		printf("Addr %d value: %d version: %d\n", i, host_data[i].value[j], host_data[i].version[j]);
	  	}
	  	printf("\n");
	}

	return result;
}
*/