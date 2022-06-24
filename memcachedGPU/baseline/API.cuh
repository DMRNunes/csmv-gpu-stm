#ifndef STM_API_H
#define STM_API_H

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>

#ifndef MaxWriteSetSize
#define MaxWriteSetSize 4
#endif // MaxWriteSetSize

#ifndef MaxReadSetSize
#define MaxReadSetSize 1024
#endif // MaxReadSetSize

#ifndef TXRecordSize
#define TXRecordSize     1048576
#define TXRecordSizeMask 1048575
#define TXRecordSizeBits 20
#endif // TXRecordSize

#ifndef MaxVersions
#define MaxVersions     16
#define MaxVersionsMask 15
#define MaxVersionsBits  4
#endif // MaxVersions

//#define threads_per_block	256

// #define bufDec(x)	((x-1+MaxVersions) % MaxVersions)
#define bufDec(x)	((x-1) & MaxVersionsMask)
#define bufInc(x)	((x+1) & MaxVersionsMask)
#define advance_pointer(x) ((x+1) & TXRecordSizeMask) 	
// #define decrease_pointer(x) (x-1+TXRecordSize) % TXRecordSize // does not need the + with the &
#define decrease_pointer(x) ((x-1) & TXRecordSizeMask)

typedef struct globaldata_
{
	volatile short head_ptr;
	volatile short tail_ptr;
	volatile int value[MaxVersions];
	volatile int version[MaxVersions];
} VertionedDataItem;

typedef struct metadata_
{
	volatile int globalClock;
	volatile int tp;
	volatile int r_hp;
	int w_hp;
} TMmetadata;

typedef struct TxRecord_
{
	volatile int transactionNumber;
	volatile int n_writes;
	volatile VertionedDataItem* writeSet[MaxWriteSetSize];
	volatile bool recordCommitted;
} TXRecord;

typedef struct readSet_
{
	int size;
	VertionedDataItem* addrs[MaxReadSetSize];
	//int version[MaxReadSetSize];
} readSet;

typedef struct writeSet_
{
	int size;
	VertionedDataItem* addrs[MaxWriteSetSize];
	int value[MaxWriteSetSize];
} writeSet;

typedef struct local_metadata_
{
	int timestamp;
	readSet  rs;
	writeSet ws;
	bool isAborted;
} local_metadata;

typedef struct Statistics_
{
	int nbCommits;
	int nbAbortsDataAge;
	int nbAbortsRecordAge;
	int nbAbortsReadWrite;
	int nbAbortsWriteWrite;
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
	long int comparisons;
	long int nbReadOnly;
	long int nbUpdates;	
} time_rate;

cudaError_t TXInit(TXRecord** d_records, TMmetadata** d_metadata);

void TXEnd(int dataSize, VertionedDataItem** host_data, VertionedDataItem** d_data, TXRecord** d_records, TMmetadata** d_metadata);

__device__ void TXBegin(TMmetadata metadata, local_metadata* txData);

__device__ bool TXCommit(int tid, TXRecord* TxRecords, TMmetadata* metadata, local_metadata txData, Statistics* stats, time_rate* times);

//__device__ bool TXWrite(VertionedDataItem* data, int value, int addr, int gaddr, local_metadata* txData);
__device__ bool TXWrite(VertionedDataItem* addr, int value, local_metadata* txData);

//__device__ int TXRead(VertionedDataItem* data, int addr, int gaddr, local_metadata* txData)
__device__ int TXRead(VertionedDataItem* addr, local_metadata* txData);

__device__ int TXReadOnly(VertionedDataItem* addr, local_metadata* txData);

#endif
