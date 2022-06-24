#include "API.cuh"

__device__ int TXRead(VertionedDataItem* addr, local_metadata* txData)
{
	int value_read;

	for(int i=0; i<txData->ws.size; i++)
	{
		if(addr == txData->ws.addrs[i])
		{
			value_read = txData->ws.value[i];
			txData->rs.addrs[txData->rs.size++]=addr;

			return value_read;
		}
	}

	if(txData->timestamp < addr->version[addr->tail_ptr])
	{
		txData->isAborted=true;
		return 0;
	}

	for(int i=addr->head_ptr;; i=bufDec(i))
	{
		if(addr->version[i] <= txData->timestamp)
		{
			value_read = addr->value[i];
			txData->rs.addrs[txData->rs.size++] = addr;	
			return value_read;
		}

		if(i == addr->tail_ptr)
			break;
	}

	txData->isAborted=true;
	return 0;
}

__device__ int TXReadOnly(VertionedDataItem* addr, local_metadata* txData)
{
	int value_read;

	if(txData->timestamp < addr->version[addr->tail_ptr])
	{
		txData->isAborted=true;
		return 0;
	}

	for(int i=addr->head_ptr;; i=bufDec(i))
	{
		if(addr->version[i] <= txData->timestamp)
		{
				value_read = addr->value[i];
				return value_read;
		}
		if(i == addr->tail_ptr)
			break;
	}

	txData->isAborted=true;
	return 0;
}

__device__ bool TXWrite(VertionedDataItem* addr, int value, local_metadata* txData)
{
	for(int i = 0; i < txData->ws.size; i++)
	{
		if(addr == txData->ws.addrs[i])
		{
			txData->ws.value[i] = value;
			return true;
		}
	}
	txData->ws.addrs[txData->ws.size] = addr;
	txData->ws.value[txData->ws.size] = value;
	txData->ws.size++;

	return true;
}

__device__ bool TXValidate(
	int         timestamp,
	TXRecord   *TxRecords,
	readSet     read_log,
	writeSet    write_log,
	int         hp,
	int         tp,
	int         oldhp,
	Statistics *stats,
	int 		tid,
	time_rate	*times
) {
	
	long int comparisons=0;
	int aux;
	while((aux=TxRecords[tp].transactionNumber)>TxRecords[hp].transactionNumber)
		tp=advance_pointer(tp);
	
	if(aux > timestamp)
	{
		atomicAdd(&(stats->nbAbortsRecordAge), 1);
		return false;
	}

	//validate read set
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
					return false; // return only when all the work is done
				}
			}
		}
	}
	times[tid].comparisons += comparisons;
	return true;	
}


__device__ bool TXValidateCollaborative(
	int         laneId,     /* different */
	int         id,         /* same */
	int         timestamp,  /* same */
	TXRecord   *TxRecords,  /* shared */
	readSet     read_log,   /* private */
	writeSet    write_log,  /* private */
	int         hp,         /* same */
	int         tp,         /* same */
	int         oldhp,      /* same */
	Statistics *stats       /* shared */
) {

	// if laneId == id & 0x1ff --> it is my transaction
	int checkTs[32];
	checkTs[laneId] = 1;

	//validate read set
	for (int i = hp; i != oldhp; i = decrease_pointer(i))
	{
		if(TxRecords[i].transactionNumber <= timestamp) break;

		for(int j = 0; j < TxRecords[i].n_writes; j++)
		{
			for (int k = 0; k < read_log.size; k++)
			{
				if(read_log.addrs[k] == TxRecords[i].writeSet[j])
				{
//printf("t%d: validation failed: read-write conflict on addr:%d\n", id, read_log.addrs[k]);
//					atomicAdd(&(stats->nbAbortsReadWrite), 1);
					return false; // return only when all the work is done
				}
			}
		}
	}

	//check for consecutive writes to the same position by transactions that are still committing
	for (int i = hp; i != oldhp; i=decrease_pointer(i))
	{
		if((TxRecords[i].recordCommitted) || (TxRecords[tp].transactionNumber <= timestamp))
			break;

		for(int j=0; j < TxRecords[i].n_writes; j++)
		{
			for (int k = 0; k < write_log.size; k++)
			{
				if(write_log.addrs[k] == TxRecords[i].writeSet[j])
				{
//printf("t%d: validation failed: write-write conflict on addr:%d\n", id, read_log.addrs[k]);
//					atomicAdd(&(stats->nbAbortsWriteWrite), 1);				
					return false;
				}
			}
		}
	}

	return true;	
}


__device__ void TXWriteBack(TMmetadata* metadata, int newtimestamp, writeSet write_log)
{

	for (int i = 0; i < write_log.size; i++)
	{
		//if(++data[write_log.addrs[i]].head_ptr == MaxVersions)
		//	data[write_log.addrs[i]].head_ptr=0;
		if (bufInc(write_log.addrs[i]->head_ptr) == write_log.addrs[i]->tail_ptr)
			write_log.addrs[i]->tail_ptr = bufInc(write_log.addrs[i]->tail_ptr);
	}
	__threadfence();

	for (int i = 0; i < write_log.size; i++)
	{
		write_log.addrs[i]->version[ bufInc(write_log.addrs[i]->head_ptr) ] = newtimestamp;
		write_log.addrs[i]->value  [ bufInc(write_log.addrs[i]->head_ptr) ] = write_log.value[i];
		//printf("t%d: %d in %d\n", id, write_log.value[i], write_log.addrs[i]);
	}
	__threadfence();

	for (int i = 0; i < write_log.size; i++)
	{
		write_log.addrs[i]->head_ptr=bufInc(write_log.addrs[i]->head_ptr);
	}
	
	// TODO: why this wait phase?
	while (newtimestamp != metadata->globalClock + 1);

	__threadfence();

	metadata->globalClock++;
}

//returns commit timestamp
__device__ int TXAddToRecordNaive(
	int                tid,
	TXRecord          *TxRecords,
	TMmetadata        *metadata,
	int                timestamp,
	readSet            read_log,
	writeSet           write_log,
	//VertionedDataItem *data,
	Statistics        *stats,
	time_rate		  *times
) {

	// with a bounded data structure that can eject TxRecords unpredictably (due to lack of space) the first thing to check
	// is if the  TX being validated has a smaller timestamp than the oldest timestamp in the TxRecords => if so we have to abort to be on the safe side

	//performance metrics
	long long int start_val1, stop_val1;
	long long int start_val2, stop_val2, totVal2=0;
	long long int start_rec, stop_rec;
	long long int start_wb, stop_wb;

	int curr_r_hp = metadata->r_hp;
	int curr_w_hp = metadata->w_hp;
	int old_r_hp = curr_r_hp;
	int newtimestamp;
	
	start_val1 = clock64();
	if(!TXValidate(timestamp, TxRecords, read_log, write_log, curr_r_hp, metadata->tp, metadata->tp, stats, tid, times))
		return 0;
	stop_val1 = clock64();

	bool isSet=false;
	start_rec = clock64();
	do
	{
		curr_r_hp = metadata->r_hp;
		curr_w_hp = metadata->w_hp;

		if(curr_r_hp != old_r_hp)
		{
			start_val2 = clock64();
			if(TXValidate(timestamp, TxRecords, read_log, write_log, curr_r_hp, metadata->tp, old_r_hp, stats, tid, times)==false)
			{	
				return 0;
			}
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
			
			//advance the tail pointer, so that 
			if(curr_w_hp == metadata->tp)
				metadata->tp = advance_pointer(metadata->tp);
			__threadfence();

			//write the WS in position curr_w_hp
			newtimestamp = TxRecords[decrease_pointer(curr_w_hp)].transactionNumber+1;
			TxRecords[curr_w_hp].transactionNumber = newtimestamp;
			TxRecords[curr_w_hp].n_writes = write_log.size;
			for(int i=0; i < write_log.size; i++)
			{
				TxRecords[curr_w_hp].writeSet[i] = write_log.addrs[i];
			}
			__threadfence();
			metadata->r_hp = advance_pointer(metadata->r_hp);
			stop_rec = clock64();

			start_wb = clock64();
			TXWriteBack(metadata, newtimestamp, write_log);
			stop_wb = clock64();

		}
	}while(!isSet);

	times[tid].val1 		+= stop_val1 - start_val1;
	times[tid].val2			+= totVal2;
	times[tid].recordWrite 	+= (stop_rec - start_rec) - totVal2;
	times[tid].dataWrite 	+= stop_wb - start_wb;
	
	return 1;			
}

__device__ bool TXCommit(int tid, TXRecord* TxRecords, TMmetadata* metadata, local_metadata txData,
							Statistics* stats, time_rate* times)
{	
	if(txData.ws.size==0)
	{	
		return true;
	}

	int newtimestamp;
	// int lanesActive = __ballot_sync(0xffffffff, 1);

	// if (lanesActive == 0xffffffff)
	// 	newtimestamp = TXAddToRecordCollaborative(id, TxRecords, metadata, txData.timestamp, txData.rs, txData.ws, data, stats);
	// else
		newtimestamp = TXAddToRecordNaive(tid, TxRecords, metadata, txData.timestamp, txData.rs, txData.ws, stats, times);
	
	return newtimestamp != 0;
}


__device__ void TXBegin(TMmetadata metadata, local_metadata* txData) //int* timestamp, readSet* read_log, writeSet* write_log)
{
	txData->timestamp = metadata.globalClock;
	txData->rs.size=0;
	txData->ws.size=0;
	txData->isAborted=false;
}

cudaError_t TXInit(TXRecord** d_records, TMmetadata** d_metadata)
{
	//GlobalData* dev_data;
//	VertionedDataItem*	h_data;
	TXRecord* 			h_records;
	TMmetadata*			h_metadata;

	cudaError_t result;
//	result = cudaMalloc((void **)d_data, dataSize*sizeof(VertionedDataItem));
//	if(result != cudaSuccess) fprintf(stderr, "Failed to allocate d_data: %s\n", cudaGetErrorString(result));
	result = cudaMalloc((void **)d_records,  TXRecordSize*sizeof(TXRecord));
	if(result != cudaSuccess) fprintf(stderr, "Failed to allocate d_records: %s\n", cudaGetErrorString(result));
	result = cudaMalloc((void **)d_metadata, sizeof(TMmetadata));
	if(result != cudaSuccess) fprintf(stderr, "Failed to allocate d_metadata: %s\n", cudaGetErrorString(result));

/*	h_data = (VertionedDataItem*)calloc(dataSize,sizeof(VertionedDataItem));
	for(int i=0; i<dataSize; i++)
	{
		h_data[i].head_ptr = 1;
		h_data[i].value[h_data[i].head_ptr] = dataArray[i];
		//printf("Addr %d value: %d version: %d\n", i, h_data[i].value[h_data[i].head_ptr], h_data[i].version[h_data[i].head_ptr]);
	}
*/	
	h_records  = (TXRecord*)   calloc(TXRecordSize, sizeof(TXRecord));
	h_metadata = (TMmetadata*) calloc(1, sizeof(TMmetadata));
	//*host_data = h_data;

//	cudaMemcpy(*d_data, h_data, dataSize*sizeof(VertionedDataItem), cudaMemcpyHostToDevice);
	cudaMemcpy(*d_records, h_records, TXRecordSize*sizeof(TXRecord), cudaMemcpyHostToDevice);
	cudaMemcpy(*d_metadata, h_metadata, sizeof(TMmetadata), cudaMemcpyHostToDevice);

	free(h_records);
	free(h_metadata);

	return result;
}

void TXEnd(int dataSize, VertionedDataItem** host_data, VertionedDataItem** d_data, TXRecord** d_records, TMmetadata** d_metadata)
{	
  	cudaMemcpy(*host_data, *d_data, dataSize*sizeof(VertionedDataItem), cudaMemcpyDeviceToHost);

//  	VertionedDataItem* h_data = *host_data;

  	//printf("\nFinal value\n");
/*	for(int i=0; i<dataSize; i++)
	{
	  	for(int j=0; j<MaxVersions; j++)
	  	{
	  		printf("Addr %d value: %d version: %d\n", i, h_data[i].value[j], h_data[i].version[j]);
	  	}
	  	printf("\n");
	}
*/
/*  	int total=0;
  	for(int i=0; i<dataSize;i++)
  	{
  		printf("%d: %d/%d\n", i, h_data[i].value[h_data[i].head_ptr], h_data[i].version[h_data[i].head_ptr]);
  		total += h_data[i].value[h_data[i].head_ptr];
  	}
  	printf("\ntotal: %d\n", total);
*/

	free(*host_data);
	cudaFree(*d_data);
	cudaFree(*d_records);
	cudaFree(*d_metadata);
}

