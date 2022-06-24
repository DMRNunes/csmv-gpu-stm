////////////////////
////	TS		////
////////////////////

#ifndef STM_API_H
#define STM_API_H

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <assert.h>
#include <cuda_runtime.h>

#include "../sync_lib/common.h"

#define MaxWriteSetSize		4
#define MaxReadSetSize		1024

#define TXRecordSize		8000
#define MaxVersions 		10

//#define threads_per_block	256

#define bufDec(x)	(x-1+MaxVersions) % MaxVersions
#define bufInc(x)	(x+1) % MaxVersions
#define advance_pointer(x) (x+1) % TXRecordSize
#define advance_pointer_nb(x,y) (x+y) % TXRecordSize	
#define decrease_pointer(x) (x-1+TXRecordSize) % TXRecordSize
#define decrease_pointer_nb(x,y) (x-y+TXRecordSize) % TXRecordSize

#define CUDA_CHECK_ERROR(func, msg) ({ \
	cudaError_t cudaError; \
	if (cudaSuccess != (cudaError = func)) { \
		fprintf(stderr, #func ": in " __FILE__ ":%i : " msg "\n   > %s\n", \
		__LINE__, cudaGetErrorString(cudaError)); \
    *((int*)0x0) = 0; /* exit(-1); */ \
	} \
  cudaError; \
})

#define RAND_R_FNC(seed) ({ \
    uint64_t next = seed; \
    uint64_t result; \
    next *= 1103515245; \
    next += 12345; \
    result = (uint64_t) (next / 65536) % 2048; \
    next *= 1103515245; \
    next += 12345; \
    result <<= 10; \
    result ^= (uint64_t) (next / 65536) % 1024; \
    next *= 1103515245; \
    next += 12345; \
    result <<= 10; \
    result ^= (uint64_t) (next / 65536) % 1024; \
    seed = next; \
    result; \
})


typedef struct globaldata_
{
	volatile ushort head_ptr;
	volatile ushort tail_ptr;
	volatile int value[MaxVersions];
	volatile uint version[MaxVersions];
} VertionedDataItem;

typedef struct metadata_
{
	ushort tp;
	ushort r_hp;
	int w_hp;
	bool hasWrapped;
} TMmetadata;

typedef struct TxRecord_
{
	//uint transactionNumber;
	ushort n_writes;
	VertionedDataItem* writeSet[MaxWriteSetSize];
	//bool recordCommitted;
} TXRecord;

typedef struct readSet_
{
	ushort size;
	VertionedDataItem* addrs[MaxReadSetSize];
} readSet;

typedef struct writeSet_
{
	ushort size;
	VertionedDataItem* addrs[MaxWriteSetSize];
	int value[MaxWriteSetSize];
} writeSet;

typedef struct scoreboard_
{
	uint volatile valid_entry;
	uint volatile lane_result[32];
} warpResult;

typedef struct Statistics_
{
	int nbCommits;
	int nbAbortsRecordAge;
	int nbAbortsReadWrite;
	int nbAbortsPreValid;
	int nbAbortsDataAge;
} Statistics;

typedef struct times_
{
	long long int total;
	long long int runtime;
	long long int commit;
	long long int dataWrite;
	long long int val1;
	long long int val2;
	long long int recordWrite;
	long long int wait;
	long long int preValidation;
	long int comparisons;
	long int nbReadOnly;
	long int nbUpdates;
} time_rate;

cudaError_t TXInit(uint threadNum, TXRecord** d_records, readSet** d_rs, writeSet** d_ws, warpResult** d_wRes);

void TXEnd(readSet** d_rs, writeSet** d_ws, warpResult** d_wRes);

__device__ uint TXBegin(uint tid, writeSet* ws, readSet* rs);

__device__ bool TXWrite(VertionedDataItem* addr, int value, writeSet* ws, uint tid);

__device__ int TXRead(VertionedDataItem* addr, uint timestamp, readSet* rs, writeSet* ws, uint tid, bool* isAborted);

__device__ int TXReadOnly(VertionedDataItem* addr, uint timestamp, readSet* rs, writeSet* ws, uint tid, bool* isAborted);

__device__ int TXAddToRecord(TMmetadata* metadata, uint* txNumber, TXRecord* TxRecords, readSet* read_log, writeSet* write_log, Statistics* stats, time_rate* times, int timestamp, int tid);

__device__ void TXWriteBack(int newtimestamp, writeSet write_log);

__device__ bool TXPreValidation(uint tid, readSet *rs, writeSet *ws);



#endif
