#include "API.cuh"
__device__ volatile uint globalClock=0;
//__device__ volatile uint attemptGC=0;

__device__ int TXRead(VertionedDataItem* data, int addr, local_metadata *txData, readSet* rs, writeSet* ws, uint tid)
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

	if(txData->timestamp < data[addr].version[data[addr].tail_ptr])
	{
		txData->isAborted=true;
		return 0;
	}

	for(int i=data[addr].head_ptr;; i=bufDec(i))
	{
		if(data[addr].version[i] <= txData->timestamp)
		{
				value_read = data[addr].value[i];
				rs[tid].addrs[rs[tid].size++] = addr;
				return value_read;
		}

		if(i == data[addr].tail_ptr)
			break;
	}

	txData->isAborted=true;
	return 0;
}

__device__ int TXReadOnly(VertionedDataItem* data, int addr, local_metadata *txData, readSet* rs, uint tid)
{
	int value_read;

	if(txData->timestamp < data[addr].version[data[addr].tail_ptr])
	{
		txData->isAborted=true;
		return 0;
	}

	for(int i=data[addr].head_ptr;; i=bufDec(i))
	{
		if(data[addr].version[i] <= txData->timestamp)
		{
				value_read = data[addr].value[i];
				return value_read;
		}

		if(i == data[addr].tail_ptr)
			break;
	}

	txData->isAborted=true;
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


__device__ bool TXValidate(TMmetadata *metadata, int timestamp, TXRecord* TxRecords, readSet read_log, writeSet write_log, int hp, int tp, int oldhp, Statistics* stats, int tid, time_rate* times)
{
	long int comparisons=0;
	int aux;

	while( (aux=TxRecords[tp].transactionNumber) > TxRecords[hp].transactionNumber)
		tp=advance_pointer(tp);

	if(aux > timestamp)
	{
		atomicAdd(&(stats->nbAbortsRecordAge), 1);
		return false;
	}

	//validate read set
	//__threadfence();
	for (int i = hp; i != oldhp; i=decrease_pointer(i))
	{
		if(TxRecords[i].transactionNumber <= timestamp)
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

//	do{
//		readAGC = attemptGC;
//		if(newtimestamp > readAGC)
//			atomicCAS((uint*)&attemptGC, readAGC, newtimestamp);
//	}
//	while(newtimestamp > attemptGC);

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

	//printf("t%d: reached GC increase\n", tid);
	
	bool isSet=false;
	do
	{
		if(isSet = (newtimestamp == globalClock+1))
		{
			globalClock++;
		}
	}
	while(!isSet);

	//printf("t%d: passed GC increase\n",tid);

//	readGC = globalClock;
//	if(attemptGC == newtimestamp)
//		atomicCAS((uint*)&globalClock, readGC, newtimestamp);
}

//returns commit timestamp
__device__ int TXAddToRecord(TMmetadata* metadata, TXRecord* TxRecords, VertionedDataItem* data, readSet* read_log, writeSet* write_log, Statistics* stats, time_rate* times, int timestamp, int tid)
{
	// with a bounded data structure that can eject TxRecords unpredictably (due to lack of space) the first thing to check
	// is if the  TX being validated has a smaller timestamp than the oldest timestamp in the TxRecords => if so we have to abort to be on the safe side
	long long int start_recWrite, stop_recWrite;
	long long int start_val1, stop_val1;
	long long int start_val2, stop_val2, totVal2=0;
	long long int start_writeback, stop_writeback;

	int curr_r_hp = metadata->r_hp;
	int curr_w_hp = metadata->w_hp;
	int old_r_hp = curr_r_hp;
	int newtimestamp;

	if(timestamp==-1)
		return 0;

	start_val1 = clock64();
	if(!TXValidate(metadata, timestamp, TxRecords, read_log[tid], write_log[tid], curr_r_hp, metadata->tp, metadata->tp, stats, tid, times))
	{
		return 0;
	}
	else
		timestamp=TxRecords[curr_r_hp].transactionNumber;
	stop_val1 = clock64();

	start_recWrite = clock64();
	bool isSet=false;
	do
	{
		curr_r_hp = metadata->r_hp;
		curr_w_hp = metadata->w_hp;

		if(curr_r_hp != old_r_hp)
		{
			start_val2 = clock64();
			if(!TXValidate(metadata, timestamp, TxRecords, read_log[tid], write_log[tid], curr_r_hp, metadata->tp, old_r_hp, stats, tid, times))
			{	
				return 0;
			}
			else
				timestamp=TxRecords[curr_r_hp].transactionNumber;
			stop_val2 = clock64();
			totVal2 += stop_val2 - start_val2;

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
			
			//advance the tail pointer before replacing the values, so that validating transactions do not access transitory data
			if(curr_w_hp == metadata->tp)
				metadata->tp = advance_pointer(metadata->tp);
			__threadfence();

			//write the WS in position curr_w_hp
			newtimestamp=TxRecords[decrease_pointer(curr_w_hp)].transactionNumber+1;
			TxRecords[curr_w_hp].transactionNumber = newtimestamp;
			TxRecords[curr_w_hp].n_writes = write_log[tid].size;
			for(int i=0; i < write_log[tid].size; i++)
			{
				TxRecords[curr_w_hp].writeSet[i] = write_log[tid].addrs[i];
			}
			__threadfence();
			metadata->r_hp = advance_pointer(metadata->r_hp);
			stop_recWrite=clock64();

			start_writeback 	= clock64();
			TXWriteBack(newtimestamp, data, write_log[tid]);
			stop_writeback 		= clock64();
		}
	}while(!isSet);

	times[tid].dataWrite+= stop_writeback - start_writeback;
	times[tid].val1  += stop_val1 - start_val1;
	times[tid].val2  += totVal2;
	times[tid].recordWrite += (stop_recWrite - start_recWrite)-totVal2;

	return newtimestamp;			
}


__device__ void TXBegin(uint tid, writeSet* ws, readSet* rs, local_metadata *txData)
{
	rs[tid].size=0;
	ws[tid].size=0;
	
	txData->valid_msg = 1;
	txData->saved_write_ptr = 0xFFFFFFFF;

	while(txData->prevTs > globalClock);
/*	{
		if(threadIdx.x==0) {
			printf("W%d is stuck %d, GC %d while I committed %d before this\n", tid/32, i, globalClock, txData->prevTs);
			i++;
		}
	}
*/	txData->timestamp = globalClock;
}

cudaError_t TXInit(int* dataArray, uint dataSize, uint threadNum, VertionedDataItem** host_data, VertionedDataItem** d_data, readSet** d_rs, writeSet** d_ws, warpResult** d_wRes)
{
	VertionedDataItem*	h_data;
	writeSet*			h_ws;
	readSet*			h_rs;
	warpResult*			h_wRes;

	cudaError_t result;
	result = cudaMalloc((void **)d_data, dataSize*sizeof(VertionedDataItem));
	if(result != cudaSuccess) fprintf(stderr, "Failed to allocate d_data: %s\n", cudaGetErrorString(result));
	result = cudaMalloc((void **)d_rs, threadNum*sizeof(readSet));
	if(result != cudaSuccess) fprintf(stderr, "Failed to allocate d_metadata: %s\n", cudaGetErrorString(result));
	result = cudaMalloc((void **)d_ws, threadNum*sizeof(writeSet));
	if(result != cudaSuccess) fprintf(stderr, "Failed to allocate d_metadata: %s\n", cudaGetErrorString(result));
	result = cudaMalloc((void **)d_wRes, threadNum/32*sizeof(warpResult));
	if(result != cudaSuccess) fprintf(stderr, "Failed to allocate wRes: %s\n", cudaGetErrorString(result));

	h_data = (VertionedDataItem*)calloc(dataSize,sizeof(VertionedDataItem));
	for(int i=0; i<dataSize; i++)
	{
		h_data[i].head_ptr = 1;
		h_data[i].value[h_data[i].head_ptr] = dataArray[i];
	}
	
	*host_data = h_data;
	h_ws       = (writeSet*) calloc(threadNum, sizeof(writeSet));
	h_rs       = (readSet*)  calloc(threadNum, sizeof(readSet));
	h_wRes     = (warpResult*)   calloc(threadNum/32, sizeof(warpResult));

	cudaMemcpy(*d_data, h_data, dataSize*sizeof(VertionedDataItem), cudaMemcpyHostToDevice);
	cudaMemcpy(*d_ws, h_ws, threadNum*sizeof(writeSet), cudaMemcpyHostToDevice);
	cudaMemcpy(*d_rs, h_rs, threadNum*sizeof(readSet), cudaMemcpyHostToDevice);
	cudaMemcpy(*d_wRes, h_wRes, threadNum/32*sizeof(warpResult), cudaMemcpyHostToDevice);


	free(h_ws);
	free(h_rs);
	free(h_wRes);

	return result;
}

void TXEnd(int dataSize, VertionedDataItem* host_data, VertionedDataItem** d_data, readSet** d_rs, writeSet** d_ws, warpResult** d_wRes)
{	
  	// cudaMemcpy(host_data, *d_data, dataSize*sizeof(VertionedDataItem), cudaMemcpyDeviceToHost);
  	
  	// long total=0;
  	// for(int i=0; i<dataSize; i++)
  	// {
  		
  	// 	printf("Item: %d hp: %d tp %d\n", i, host_data[i].head_ptr, host_data[i].tail_ptr);
  	// 	for(int j=0; j<MaxVersions; j++)
  	// 		printf("\t%d\t%d\n", host_data[i].version[j], host_data[i].value[j]);
  		
  	// 	total += (long)host_data[i].value[host_data[i].head_ptr];
  	// }
  	// if(total != 100*dataSize)
  	// 	printf("Consistency fail: Total %d\n", total);
  	//if(total != dataSize*1000)
  	//	printf("Invariance not maintanted: total bank amount is %d when it should be %d\n", total, dataSize*1000);

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