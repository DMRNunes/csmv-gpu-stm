#include "API.cuh"

__device__ int TXRead(VertionedDataItem* data, int addr, local_metadata* txData)
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
			txData->rs.addrs[txData->rs.size++] = addr;				
			return value_read;
		}

		if(i == data[addr].tail_ptr)
			break;
	}

	txData->isAborted=true;
	return 0;
}

__device__ int TXReadOnly(VertionedDataItem* data, int addr, local_metadata* txData)
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

__device__ bool TXWrite(VertionedDataItem* data, int value, int addr, local_metadata* txData)
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
					times[tid].comparisons += comparisons;
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
					atomicAdd(&(stats->nbAbortsReadWrite), 1);
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
					atomicAdd(&(stats->nbAbortsWriteWrite), 1);				
					return false;
				}
			}
		}
	}

	return true;	
}

__device__ void TXWriteBack(TMmetadata* metadata, int newtimestamp, VertionedDataItem* data, writeSet write_log)
{
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
	
	// TODO: why this wait phase?
	while (newtimestamp != metadata->globalClock + 1);

	__threadfence();

	metadata->globalClock++;
}


//returns commit timestamp
__device__ int TXAddToRecordNaive(
	int               tid,
	TXRecord          *TxRecords,
	TMmetadata        *metadata,
	int               timestamp,
	readSet           read_log,
	writeSet          write_log,
	VertionedDataItem *data,
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
			TXWriteBack(metadata, newtimestamp, data, write_log);
			stop_wb = clock64();
		}
	}while(!isSet);

	times[tid].val1 		+= stop_val1 - start_val1;
	times[tid].val2			+= totVal2;
	times[tid].recordWrite 	+= (stop_rec - start_rec) - totVal2;
	times[tid].dataWrite 	+= stop_wb - start_wb;

	return 1;			
}

//returns commit timestamp
__device__ int TXAddToRecordCollaborative(
	int                id,
	TXRecord          *TxRecords,
	TMmetadata        *metadata,
	int                timestamp,
	readSet            read_log,
	writeSet           write_log,
	VertionedDataItem *data,
	Statistics        *stats
) {

	// with a bounded data structure that can eject TxRecords unpredictably (due to lack of space) the first thing to check
	// is if the  TX being validated has a smaller timestamp than the oldest timestamp in the TxRecords => if so we have to abort to be on the safe side

	int curr_r_hp = metadata->r_hp;
	int curr_w_hp = metadata->w_hp;
	int old_r_hp = curr_r_hp;
	int myLaneId = threadIdx.x & 0x1ff;
	int TXid = id & (~0x1ff);
	int resEachLane[32];

	// TODO: check if they are divergent
	int lanesActive = __ballot_sync(0xffffffff, 1);
	if (lanesActive != 0xffffffff && id % 32 == 0) printf("[%i] some bit disabled: %x\n", id, lanesActive);

	resEachLane[myLaneId] = 1;

	if(TxRecords[metadata->tp].transactionNumber > timestamp)
	{
		atomicAdd(&(stats->nbAbortsRecordAge), 1);
		resEachLane[myLaneId] = 0; // all aborted TXs reported here
	}

	__syncwarp(); // in pascal this should be ignored

	// assuming each lane has a TX to validate, there are 32
	int currTXid = TXid;
	for (int i = 0; i < 32; ++i) {
		if (resEachLane[i] == 0) continue; /* timestamp validation failed */
		
		// if (!TXValidateCollaborative(
		// 	myLaneId,
		// 	currTXid,
		// 	timestamp,
		// 	TxRecords,
		// 	read_log,
		// 	write_log,
		// 	curr_r_hp,
		// 	metadata->tp,
		// 	metadata->tp,
		// 	stats) /* every one gets the same result */
		// ) { /* aborted */ resEachLane[i] = 0; continue; }

		bool isSet = false;

		// we may need to repeat this transaction as other warps may be working on it
		do {
			curr_r_hp = metadata->r_hp;
			curr_w_hp = metadata->w_hp;

			if (curr_r_hp != old_r_hp) {
				if (!TXValidateCollaborative(
					myLaneId,
					currTXid,
					timestamp,
					TxRecords,
					read_log, 
					write_log,
					curr_r_hp,
					metadata->tp,
					old_r_hp,
					stats)
				) { 
					resEachLane[i] = 0; break;
				}
				old_r_hp = curr_r_hp;
			}

			// other warp is doing this
			if (curr_r_hp != curr_w_hp) { continue; }

			if (isSet = (atomicCAS(&(metadata->w_hp), curr_w_hp, advance_pointer(curr_w_hp)) == curr_w_hp))
			{
				// if (++curr_w_hp == TXRecordSize) curr_w_hp = 0;
				curr_w_hp = (curr_w_hp + 1) & TXRecordSizeMask;

				// if the transaction to be replaced has not yet finish writting
				// to the global memory, spin until it does
				if (TxRecords[curr_w_hp].transactionNumber > 0)
				{
					while (TxRecords[curr_w_hp].recordCommitted == false);	
				}	
				
				// advance the tail pointer, so that other threads can use the queue
				if (curr_w_hp == metadata->tp) metadata->tp = advance_pointer(metadata->tp);
				
				__threadfence(); // transaction pointer visible to other blocks

				//write the WS in position curr_w_hp
				TxRecords[curr_w_hp].transactionNumber = TxRecords[decrease_pointer(curr_w_hp)].transactionNumber + 1;
				TxRecords[curr_w_hp].n_writes = write_log.size;
				TxRecords[curr_w_hp].recordCommitted = false;

				// moves tx write-set to globally visible record
				for (int i = 0; i < write_log.size; i++)
				{
					TxRecords[curr_w_hp].writeSet[i] = write_log.addrs[i];
					//printf("t%d: wrote %d on %d\n", id, write_log.value[i], write_log.addrs[i]);
				}

				__threadfence();

				metadata->r_hp = advance_pointer(metadata->r_hp);
	//			printf("t%d: updated r_hp\n", id);

				//TXWriteBack(TxRecords, metadata, TxRecords[curr_w_hp].transactionNumber, write_log, id);

			}
			// else: lost the CAS, retry
		} while(!isSet);

		old_r_hp = curr_r_hp;

		currTXid++;
	}
	

	return resEachLane[myLaneId];			
}

__device__ bool TXCommit(int id, TXRecord* TxRecords, VertionedDataItem* data, TMmetadata* metadata, local_metadata txData, Statistics *stats, time_rate *times)
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
		newtimestamp = TXAddToRecordNaive(id, TxRecords, metadata, txData.timestamp, txData.rs, txData.ws, data, stats, times);
	
	return newtimestamp != 0;
}


__device__ void TXBegin(TMmetadata metadata, local_metadata* txData) //int* timestamp, readSet* read_log, writeSet* write_log)
{
	txData->timestamp = metadata.globalClock;
	txData->rs.size=0;
	txData->ws.size=0;
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

void TXEnd(int dataSize, VertionedDataItem* host_data, VertionedDataItem** d_data, TXRecord** d_records, TMmetadata** d_metadata)
{	
	cudaMemcpy(host_data, *d_data, dataSize*sizeof(VertionedDataItem), cudaMemcpyDeviceToHost);
  	
  	long total=0;
  	for(int i=0; i<dataSize; i++)
  	{
  		/*
  		printf("Item: %d hp: %d tp %d\n", i, host_data[i].head_ptr, host_data[i].tail_ptr);
  		for(int j=0; j<MaxVersions; j++)
  			printf("\t%d\t%d\n", host_data[i].version[j], host_data[i].value[j]);
  		*/
  		total += (long)host_data[i].value[host_data[i].head_ptr];
  	}
  	if(total != 10*dataSize)
  		printf("Consistency fail: Total %d\n", total);
  	//if(total != dataSize*1000)
  	//	printf("Invariance not maintanted: total bank amount is %d when it should be %d\n", total, dataSize*1000);

	free(host_data);
	cudaFree(*d_data);
	cudaFree(*d_records);
	cudaFree(*d_metadata);
}

