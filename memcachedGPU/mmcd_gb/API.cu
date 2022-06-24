//EXPERIMENTAL VERSION
////////////////////
////	GB		////
////////////////////

#include "API.cuh"
__device__ volatile uint globalClock=0;
//__device__ volatile uint attemptGC=0;

enum status {TA, A, C};

__device__ int TXRead(VertionedDataItem* addr, uint timestamp, readSet* rs, writeSet* ws, uint tid, bool* isAborted)
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

	if(timestamp < addr->version[addr->tail_ptr])
	{
		*isAborted=true;
		return 0;
	}

	for(int i=addr->head_ptr;; i=bufDec(i))
	{
		if(addr->version[i] <= timestamp)
		{
				value_read = addr->value[i];
				rs[tid].addrs[rs[tid].size++] = addr;
				return value_read;
		}

		if(i == addr->tail_ptr)
			break;
	}

	*isAborted=true;
	return 0;
}

__device__ int TXReadOnly(VertionedDataItem* addr, uint timestamp, readSet* rs, writeSet* ws, uint tid, bool* isAborted)
{
	int value_read;

	if(timestamp < addr->version[addr->tail_ptr])
	{
		*isAborted=true;
		return 0;
	}

	for(int i=addr->head_ptr;; i=bufDec(i))
	{
		if(addr->version[i] <= timestamp)
		{
				value_read = addr->value[i];
				return value_read;
		}

		if(i == addr->tail_ptr)
			break;
	}

	*isAborted=true;
	return 0;
}

__device__ bool TXWrite(VertionedDataItem* addr, int value, writeSet* ws, uint tid)
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

__device__ bool TXPreValidation(uint tid, readSet *rs, writeSet *ws)
{
	uint32_t conflict_var = 0; //valid_ballot, status_ballot;
	int status = TA;
	int other_status;
	int mylane = get_lane_id();

	for(int i=mylane; i > 0; i--)
	{
		for(int j=0; j < ws[tid-i].size; j++)
		{
			for (int k = 0; k < rs[tid].size; k++)
			{
				if( rs[tid].addrs[k] == ws[tid-i].addrs[j])
					conflict_var |=	1<<(mylane-i);
			}
		}
	}
	// All threads with no conflicts have their status set to C
	if (conflict_var == 0)
		status = C;

	__syncwarp();
	// Ballot is used to see if the ANY thread in the warp has to do the loop, starting a new round in that case
	while(__ballot_sync(__activemask(), status==TA) !=0)
	{
		
		// Iterate over the lanes
		// The +1 is necessary here so that the source lane of the shuffle is not masked off during the operation
		for(int i=0; i<warpSize; i++)
		{	
			// This shuffle is only performed by the source lane (i) and the ones with higher LaneID
			//__syncwarp();
			other_status = __shfl_sync(__activemask(), status, i);

			// We only care about the status of the source lane if we are in an inconclusive state(TA) and we have an active conflict with it
			if( (status == TA) && (conflict_var & (1<<i)) )
			{
				// if the conflicting thread is committed then we should abort
				if(other_status == C)
					status = A;
				
				// if the conflicting thread is aborted, then we eliminate the conflict by setting to 0 the corresponding bit in our conflict_var
				//		if by doing so we have no other active conflicts then we can commit
				else if(other_status == A)
				{
					conflict_var &= ~(1<<i);
					if(conflict_var == 0)
						status = C;
				}
			}
		}
		
		/*
		// All threads do a ballot to know the status of the other lanes
		//	threads whose status is NOT TA are in a final and valid state, the corresponding bit in the ballot will be set to 1 for these threads
		valid_ballot  = __ballot_sync(__activemask(), status != TA);
		//	threads that have reached a final state were they were committed (C) have the correspoding bit in the next ballot set to 1
		status_ballot = __ballot_sync(__activemask(), status == C);
		// The idea is to first check the result for the first ballot to see if the thread has or not reached a final state
		// 	and if it has indeed reached such a state, then we check the second ballot to see if it was committed (C) or aborted (A)

		for(int i=0; i<mylane; i++)
		{
			// We only care about the status of the source lane if we are in an inconclusive state(TA) and we have an active conflict with it
			if( (status == TA) && (conflict_var & (1<<i)) )
			{
				// If the conflicting thread has reached a final state
				if(valid_ballot & (1<<i))
				{				
					// IF the conflicting thread committed then we abort
					if(status_ballot & (1<<i))
						status = A;
					// otherwise, we eliminate the conflict in our conflict_var, 
					//		if by doing so we have no other active conflicts then we can commit
					else
					{
						conflict_var &= ~(1<<i);
						if(conflict_var == 0)
							status = C;
					}
				}
			}
		}
		*/
		__syncwarp();
	}

	return status==C;
}

__device__ bool TXValidate(TMmetadata *metadata, int timestamp, TXRecord* TxRecords, readSet *rs, int hp, Statistics* stats, int tid, time_rate* times)
{
	long int comparisons=0;
	ushort wid = threadIdx.x/warpSize;
	bool result = false;
	__shared__ bool conflict[32];
	__shared__ bool spurious[32];
	int mylane = get_lane_id();
	int ts;
	int anchor, nbIters;

	if(timestamp>=0)
		if(TxRecords[metadata->tp].transactionNumber > timestamp)
		{
			atomicAdd(&(stats->nbAbortsRecordAge), 1);
			timestamp=-1;
		}
	
	for(int lane=0; lane<warpSize; lane++)
	{
		ts = __shfl_sync(__activemask(), timestamp, lane);
		if(ts < 0)
			continue;
		conflict[wid]=false;
		spurious[wid]=false;

        anchor = ts % TXRecordSize;
        if (hp >= anchor)
            nbIters= ceilf((float)(hp - mylane - anchor)/(float)warpSize);
        else
            nbIters= ceilf((float)(hp + TXRecordSize - mylane - anchor)/(float)warpSize);
            
		for(int i = decrease_pointer_nb(hp,mylane); conflict[wid]==false && spurious[wid]==false && nbIters > 0; i=decrease_pointer_nb(i,warpSize), nbIters--)
		{ 
			comparisons++;
		    for(int j=0; j < TxRecords[i].n_writes; j++)
		    {
		        for(int k=0; k < rs[tid+lane-mylane].size; k++)
		        {
		            if(rs[tid+lane-mylane].addrs[k] == TxRecords[i].writeSet[j])
		            {
		                conflict[wid] = true;
		            }
		        }   
		    }
		    if (TxRecords[anchor].transactionNumber > ts) {
		        spurious[wid]=true;
		    }  
			
        }
        __syncwarp();

        if(lane == mylane)
        {
            result = !conflict[wid] && !spurious[wid];
            if(conflict[wid])
                atomicAdd(&(stats->nbAbortsReadWrite), 1);
            else if(spurious[wid])
            	atomicAdd(&(stats->nbAbortsRecordAge), 1);
        }
    }
    times[tid].comparisons += comparisons;
    return result;
}


__device__ void TXWriteBack(int newtimestamp, writeSet write_log)
{
	uint32_t wbBallot = __ballot_sync(__activemask(), newtimestamp);
	uint warpLeader = find_nth_bit(wbBallot, 0, 1);
	uint nbCommits  = count_bit(wbBallot);

	for (int i = 0; i < write_log.size; i++)
	{
		if(bufInc(write_log.addrs[i]->head_ptr) == write_log.addrs[i]->tail_ptr)
			write_log.addrs[i]->tail_ptr=bufInc(write_log.addrs[i]->tail_ptr);
	}
	__threadfence();

	for (int i = 0; i < write_log.size; i++)
	{
		write_log.addrs[i]->version[ bufInc(write_log.addrs[i]->head_ptr) ] = newtimestamp;
		write_log.addrs[i]->value  [ bufInc(write_log.addrs[i]->head_ptr) ] = write_log.value[i];
	}
	__threadfence();

	for (int i = 0; i < write_log.size; i++)
	{
		write_log.addrs[i]->head_ptr=bufInc(write_log.addrs[i]->head_ptr);
	}
	__threadfence();
	
	if(get_lane_id() == warpLeader)
	{
		while(globalClock != newtimestamp-1);
		globalClock += nbCommits;
	}
	__syncwarp();
}

//returns commit timestamp
__device__ int TXAddToRecord(TMmetadata* metadata, TXRecord* TxRecords, readSet* read_log, writeSet* write_log, Statistics* stats, time_rate* times, int timestamp, int tid)
{

	// with a bounded data structure that can eject TxRecords unpredictably (due to lack of space) the first thing to check
	// is if the  TX being validated has a smaller timestamp than the oldest timestamp in the TxRecords => if so we have to abort to be on the safe side
	long long int start_recWrite, stop_recWrite;
	long long int start_val1, stop_val1;
	long long int start_val2, stop_val2, totVal2=0;

	int curr_r_hp = metadata->r_hp;
	int curr_w_hp = metadata->w_hp;
	int old_r_hp = curr_r_hp;
	int newtimestamp, offset;
	
	int live_tx = 1;
	int insert = -1;

	start_val1 = clock64();
	if(!TXValidate(metadata, timestamp, TxRecords, read_log, curr_r_hp, stats, tid, times))
	{
		live_tx=0;
		timestamp=-1;
	}
	else
		timestamp=TxRecords[curr_r_hp].transactionNumber;
	stop_val1 = clock64();

	uint validation_ballot = vote_ballot(live_tx);
	if(validation_ballot == 0)
		return 0;

	uint warp_leader = find_nth_bit(validation_ballot, 0, 1);
	uint passed_val = count_bit(validation_ballot);

	start_recWrite = clock64();
	do
	{
		curr_r_hp = metadata->r_hp;
		curr_w_hp = metadata->w_hp;

		if(curr_r_hp != old_r_hp)
		{
			start_val2 = clock64();
			if(!TXValidate(metadata, timestamp, TxRecords, read_log, curr_r_hp, stats, tid, times))
			{
				live_tx=0;
				timestamp=-1;
			}
			else
				timestamp=TxRecords[curr_r_hp].transactionNumber;
			stop_val2 = clock64();
			totVal2 += stop_val2 - start_val2;
			
			old_r_hp = curr_r_hp;
			validation_ballot = vote_ballot(live_tx);
			if(validation_ballot == 0)
				return 0;
			warp_leader = find_nth_bit(validation_ballot, 0, 1);
			passed_val = count_bit(validation_ballot);
		}
		if(curr_r_hp != curr_w_hp)
		{
			continue;
		}
		if(get_lane_id() == warp_leader)
		{
			if(atomicCAS(&(metadata->w_hp), curr_w_hp, advance_pointer_nb(curr_w_hp, passed_val)) == curr_w_hp)
			{
				if((!metadata->hasWrapped) && (curr_w_hp > metadata->w_hp))
					metadata->hasWrapped=true;
				if(metadata->hasWrapped)
					metadata->tp = advance_pointer(metadata->w_hp);
				__threadfence();

				if(++curr_w_hp == TXRecordSize)
					curr_w_hp = 0;
				insert = curr_w_hp;
			}
		}
		__syncwarp();
		insert = __shfl_sync(__activemask(), insert, warp_leader);
	}while(insert < 0);

	//write the WS in position curr_w_hp
	if(live_tx)
	{
		offset = count_bit(set_bits(0, validation_ballot, 0, get_lane_id() ));
		newtimestamp=TxRecords[decrease_pointer(insert)].transactionNumber+offset+1;
		insert = advance_pointer_nb(insert, offset);

		TxRecords[insert].transactionNumber = newtimestamp;
		TxRecords[insert].n_writes = write_log[tid].size;
		for(int i=0; i < write_log[tid].size; i++)
		{
			TxRecords[insert].writeSet[i] = write_log[tid].addrs[i];
		}
	}
	__threadfence();

	if(get_lane_id() == warp_leader)
		metadata->r_hp = advance_pointer_nb(metadata->r_hp, count_bit(validation_ballot));

	stop_recWrite=clock64();

	if(live_tx)
	{
		times[tid].val1  += stop_val1 - start_val1;
		times[tid].val2  += totVal2;
		times[tid].recordWrite += (stop_recWrite - start_recWrite)-totVal2;
	}

	return newtimestamp;			
}


__device__ uint TXBegin(uint tid, writeSet* ws, readSet* rs)
{
	rs[tid].size=0;
	ws[tid].size=0;
	return globalClock;
}

cudaError_t TXInit(uint threadNum, TXRecord** d_records, readSet** d_rs, writeSet** d_ws, warpResult** d_wRes)
{
	TXRecord *			h_records;
	readSet*			h_rs;
	writeSet*			h_ws;
	warpResult*			h_wRes;

	cudaError_t result;
	result = cudaMalloc((void **)d_records, TXRecordSize*sizeof(TXRecord));
	if(result != cudaSuccess) fprintf(stderr, "Failed to allocate d_records: %s\n", cudaGetErrorString(result));
	result = cudaMalloc((void **)d_rs, threadNum*sizeof(readSet));
	if(result != cudaSuccess) fprintf(stderr, "Failed to allocate d_metadata: %s\n", cudaGetErrorString(result));
	result = cudaMalloc((void **)d_ws, threadNum*sizeof(writeSet));
	if(result != cudaSuccess) fprintf(stderr, "Failed to allocate d_metadata: %s\n", cudaGetErrorString(result));
	result = cudaMalloc((void **)d_wRes, threadNum/32*sizeof(warpResult));
	if(result != cudaSuccess) fprintf(stderr, "Failed to allocate wRes: %s\n", cudaGetErrorString(result));

	h_records  = (TXRecord*) calloc(TXRecordSize, sizeof(TXRecord));
	h_ws       = (writeSet*) calloc(threadNum, sizeof(writeSet));
	h_rs       = (readSet*)  calloc(threadNum, sizeof(readSet));
	h_wRes     = (warpResult*)   calloc(threadNum/32, sizeof(warpResult));

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

void TXEnd(readSet** d_rs, writeSet** d_ws, warpResult** d_wRes)
{	
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