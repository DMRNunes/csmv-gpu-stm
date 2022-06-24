/*
 * msg_aux.h
 *
 *  Created on: Jan 1, 2019
 *      Author: redudie
 */

#ifndef MSG_AUX_H_
#define MSG_AUX_H_

#include <cuda.h>
#include <inttypes.h>
#include <bitset>
#include <cmath>

#define BASE_SEND 0
#define ADV_SEND 1

typedef struct gbc_t_t {
	uint (*queue_data)[NUM_RECEIVER][GC_BUFF_SIZE];
	uint (*ready_bitmask)[GC_NUM_MASK];
	uint* global_tail_ptr;
	uint* write_ptr;
	int receiver_start;
	int receiver_end;
	uint num_sender_per_bk;
	int leader_warp_id;
	int leader_worker_warp_id;
	int* exit_counter_global;
	uint msg_size;
	uint work_buff_size;
	uint channel_id;
} gbc_t;

typedef struct gbc_pack_t_t {
	gbc_t gbc[NUM_CHANNEL];
} gbc_pack_t;

void init_gbc(gbc_t* gbc, uint msg_size, int exit_counter, int receiver_start,
		int receiver_end, uint num_sender_per_bk, int leader_warp_id,
		int leader_worker_warp_id, uint work_buff_size) {

	//assert(GC_BUFF_SIZE >= 8192);
	std::bitset<32> t3(NUM_RESET_BUF - 1);
	assert(t3.count() == AB_RESET_NUM_BIT);

	//needed for adv send to work
	assert(NUM_RECEIVER <= 32);
	assert(work_buff_size <= WORK_BUFF_SIZE_MAX);
	//check config
	assert((1 << DIV_SHIFT) == DIV_FACTOR);

	assert(msg_size <= MSG_SIZE_MAX);
	gbc->msg_size = msg_size;
	gbc->work_buff_size = work_buff_size;
	gbc->receiver_start = receiver_start;
	gbc->receiver_end = receiver_end;
	gbc->num_sender_per_bk = num_sender_per_bk;
	if (cudaMalloc((void **) &(gbc->queue_data),
			msg_size * NUM_RECEIVER * GC_BUFF_SIZE * sizeof(uint))
			!= cudaSuccess) {
		assert(false);
	}

	if (cudaMalloc((void **) &(gbc->ready_bitmask),
			NUM_RECEIVER * GC_NUM_MASK * sizeof(uint)) != cudaSuccess) {
		assert(false);
	}
	cudaMemset((uint*) (gbc->ready_bitmask), 0,
			NUM_RECEIVER * GC_NUM_MASK * sizeof(uint));

	if (cudaMalloc((void **) &(gbc->global_tail_ptr),
			NUM_RECEIVER * sizeof(uint)) != cudaSuccess) {
		assert(false);
	}
	cudaMemset(gbc->global_tail_ptr, 0, NUM_RECEIVER * sizeof(uint));

	if (cudaMalloc((void **) &(gbc->write_ptr), NUM_RECEIVER * sizeof(uint))
			!= cudaSuccess) {
		assert(false);
	}
	cudaMemset(gbc->write_ptr, 0, NUM_RECEIVER * sizeof(uint));
	gbc->leader_warp_id = leader_warp_id;
	gbc->leader_worker_warp_id = leader_worker_warp_id;
	if (cudaMalloc((void **) &(gbc->exit_counter_global), sizeof(uint))
			!= cudaSuccess) {
		assert(false);
	}
	cudaMemcpy(gbc->exit_counter_global, &exit_counter, sizeof(uint),
			cudaMemcpyHostToDevice);

}

void free_gbc(gbc_pack_t gbc_pack) {
	for (int i = 0; i < NUM_CHANNEL; i++) {
		cudaFree(gbc_pack.gbc[i].queue_data);
		cudaFree(gbc_pack.gbc[i].ready_bitmask);
		cudaFree(gbc_pack.gbc[i].global_tail_ptr);
		cudaFree(gbc_pack.gbc[i].write_ptr);
	}

}
/*
 __device__ __forceinline__ unsigned get_local_lock_id(uint data_id) {
 uint upper_id = ((data_id) >> (DIV_SHIFT + NUM_RECEIVE_SHIFT)) << DIV_SHIFT;
 uint lower_id = data_id & (DIV_FACTOR - 1);
 return (upper_id | lower_id) % NUM_LOCK;
 }
 */

__shared__ uint _offload_tail_ptr_cpy[NUM_RECEIVER];
///////
__shared__ uint _offload_coal_buf_data[MSG_SIZE_OFFLOAD][NUM_RECEIVER][64];
///////
__shared__ uint adv_tail_ptr_cpy[NUM_CHANNEL][NUM_RECEIVER];
__shared__ uint adv_coal_buf_ptrs[NUM_CHANNEL][2][NUM_RECEIVER];
__shared__ uint adv_coal_buf_state[NUM_CHANNEL][NUM_RECEIVER];
__shared__ uint adv_saved_write_ptr[NUM_CHANNEL][NUM_RECEIVER];
__shared__ uint adv_reader_states[NUM_CHANNEL][NUM_RECEIVER];
__shared__ uint adv_rr_ptr[NUM_CHANNEL];
__shared__ uint adv_last_update_time[NUM_CHANNEL];
__shared__ uint adv_update_lock[NUM_CHANNEL];
__shared__ int adv_local_exit_counter[NUM_CHANNEL];

__device__ __forceinline__ unsigned get_local_lock_id(uint data_id) {
	uint upper_id = (data_id / (DIV_FACTOR * NUM_RECEIVER)) << DIV_SHIFT;
	uint lower_id = data_id & (DIV_FACTOR - 1);
	return (upper_id | lower_id) % NUM_LOCK;
}
/*
 __device__ __forceinline__ unsigned get_local_lock_id(uint data_id) {
 return data_id % NUM_LOCK;
 }
 */
//buffers for work assignment and leader metadata
__shared__ uint work_buffer[NUM_CHANNEL][WORK_BUFF_SIZE_MAX][32];
__shared__ uint work_buffer_status[NUM_CHANNEL][32];

__shared__ uint reset_accu[NUM_CHANNEL][NUM_RESET_BUF];
__shared__ uint reset_target[NUM_CHANNEL][NUM_RESET_BUF];
__shared__ uint reset_head[NUM_CHANNEL];
__shared__ uint reset_tail[NUM_CHANNEL];
__shared__ uint local_tail_ptr[NUM_CHANNEL];

__shared__ bool leader_terminate[NUM_CHANNEL];
__shared__ uint leader_exit_check[NUM_CHANNEL];
__shared__ uint leader_head_ptr[NUM_CHANNEL];
__shared__ uint leader_new_head_ptr[NUM_CHANNEL];
__shared__ uint leader_assign_reset_buf[NUM_CHANNEL];

#if MSG_SIZE_MAX == 1
#define SEND_ARG_DEF  uint val0
#endif

#if MSG_SIZE_MAX == 2
#define SEND_ARG_DEF  uint val0, int val1
#endif

#if MSG_SIZE_MAX == 3
#define SEND_ARG_DEF  uint val0, uint val1, uint val2
#endif

#if MSG_SIZE_MAX == 4
#define SEND_ARG_DEF  uint val0, uint val1, uint val2, uint val3
#endif

#if MSG_SIZE_MAX == 5
#define SEND_ARG_DEF  uint val0, uint val1, uint val2, uint val3, uint val4
#endif

#if MSG_SIZE_MAX == 6
#define SEND_ARG_DEF  uint val0, uint val1, uint val2, uint val3, uint val4, uint val5
#endif

#if MSG_SIZE_MAX == 7
#define SEND_ARG_DEF  uint val0, uint val1, uint val2, uint val3, uint val4, uint val5, uint val6
#endif

#if MSG_SIZE_MAX == 8
#define SEND_ARG_DEF  uint val0, uint val1, uint val2, uint val3, uint val4, uint val5, uint val6, uint val7
#endif

#if MSG_SIZE_MAX == 1
#define SEND_ARG  val0
#endif

#if MSG_SIZE_MAX == 2
#define SEND_ARG  val0,  val1
#endif

#if MSG_SIZE_MAX == 3
#define SEND_ARG  val0,  val1,  val2
#endif

#if MSG_SIZE_MAX == 4
#define SEND_ARG   val0,  val1,  val2,  val3
#endif

#if MSG_SIZE_MAX == 5
#define SEND_ARG   val0,  val1,  val2,  val3,  val4
#endif

#if MSG_SIZE_MAX == 6
#define SEND_ARG   val0,  val1,  val2,  val3,  val4,  val5
#endif

#if MSG_SIZE_MAX == 7
#define SEND_ARG   val0,  val1,  val2,  val3,  val4,  val5,  val6
#endif

#if MSG_SIZE_MAX == 8
#define SEND_ARG   val0,  val1,  val2,  val3,  val4,  val5,  val6,  val7
#endif

#if MSG_SIZE_MAX == 1
#define SEND_ARG_DEF_REF  uint& val0
#endif

#if MSG_SIZE_MAX == 2
#define SEND_ARG_DEF_REF  uint& val0, uint& val1
#endif

#if MSG_SIZE_MAX == 3
#define SEND_ARG_DEF_REF  uint& val0, uint& val1, uint& val2
#endif

#if MSG_SIZE_MAX == 4
#define SEND_ARG_DEF_REF  uint& val0, uint& val1, uint& val2, uint& val3
#endif

#if MSG_SIZE_MAX == 5
#define SEND_ARG_DEF_REF  uint& val0, uint& val1, uint& val2, uint& val3, uint& val4
#endif

#if MSG_SIZE_MAX == 6
#define SEND_ARG_DEF_REF  uint& val0, uint& val1, uint& val2, uint& val3, uint& val4, uint& val5
#endif

#if MSG_SIZE_MAX == 7
#define SEND_ARG_DEF_REF  uint& val0, uint& val1, uint& val2, uint& val3, uint& val4, uint& val5, uint& val6
#endif

#if MSG_SIZE_MAX == 8
#define SEND_ARG_DEF_REF  uint& val0, uint& val1, uint& val2, uint& val3, uint& val4, uint& val5, uint& val6, uint& val7
#endif

#if MSG_SIZE_MAX == 1
#define SEND_VAR_DEF  uint val0
#endif

#if MSG_SIZE_MAX == 2
#define SEND_VAR_DEF  uint val0,  val1
#endif

#if MSG_SIZE_MAX == 3
#define SEND_VAR_DEF  uint val0,  val1,  val2
#endif

#if MSG_SIZE_MAX == 4
#define SEND_VAR_DEF  uint val0,  val1,  val2,  val3
#endif

#if MSG_SIZE_MAX == 5
#define SEND_VAR_DEF  uint val0,  val1,  val2,  val3,  val4
#endif

#if MSG_SIZE_MAX == 6
#define SEND_VAR_DEF  uint val0,  val1,  val2,  val3,  val4,  val5
#endif

#if MSG_SIZE_MAX == 7
#define SEND_VAR_DEF  uint val0,  val1,  val2,  val3,  val4,  val5,  val6
#endif

#if MSG_SIZE_MAX == 8
#define SEND_VAR_DEF  uint val0,  val1,  val2,  val3,  val4,  val5,  val6,  val7
#endif

#if MSG_SIZE_OFFLOAD == 1
#define VAR_BUF_DEF0 uint value0_buf0;
#endif

#if MSG_SIZE_OFFLOAD == 2
#define VAR_BUF_DEF0 uint value0_buf0, value1_buf0;
#endif

#if MSG_SIZE_OFFLOAD == 3
#define VAR_BUF_DEF0 uint value0_buf0, value1_buf0, value2_buf0;
#endif

#if MSG_SIZE_OFFLOAD == 4
#define VAR_BUF_DEF0 uint value0_buf0, value1_buf0, value2_buf0, value3_buf0;
#endif

#if MSG_SIZE_OFFLOAD == 5
#define VAR_BUF_DEF0 uint value0_buf0, value1_buf0, value2_buf0, value3_buf0, value4_buf0;
#endif

#if MSG_SIZE_OFFLOAD == 6
#define VAR_BUF_DEF0 uint value0_buf0, value1_buf0, value2_buf0, value3_buf0, value4_buf0, value5_buf0;
#endif

#if MSG_SIZE_OFFLOAD == 7
#define VAR_BUF_DEF0 uint value0_buf0, value1_buf0, value2_buf0, value3_buf0, value4_buf0, value5_buf0, value6_buf0;
#endif

#if MSG_SIZE_OFFLOAD == 8
#define VAR_BUF_DEF0 uint value0_buf0, value1_buf0, value2_buf0, value3_buf0, value4_buf0, value5_buf0, value6_buf0, value7_buf0;
#endif

#if MSG_SIZE_OFFLOAD == 1
#define VAR_BUF_DEF1 uint value0_buf1;
#endif

#if MSG_SIZE_OFFLOAD == 2
#define VAR_BUF_DEF1 uint value0_buf1, value1_buf1;
#endif

#if MSG_SIZE_OFFLOAD == 3
#define VAR_BUF_DEF1 uint value0_buf1, value1_buf1, value2_buf1;
#endif

#if MSG_SIZE_OFFLOAD == 4
#define VAR_BUF_DEF1 uint value0_buf1, value1_buf1, value2_buf1, value3_buf1;
#endif

#if MSG_SIZE_OFFLOAD == 5
#define VAR_BUF_DEF1 uint value0_buf1, value1_buf1, value2_buf1, value3_buf1, value4_buf1;
#endif

#if MSG_SIZE_OFFLOAD == 6
#define VAR_BUF_DEF1 uint value0_buf1, value1_buf1, value2_buf1, value3_buf1, value4_buf1, value5_buf1;
#endif

#if MSG_SIZE_OFFLOAD == 7
#define VAR_BUF_DEF1 uint value0_buf1, value1_buf1, value2_buf1, value3_buf1, value4_buf1, value5_buf1, value6_buf1;
#endif

#if MSG_SIZE_OFFLOAD == 8
#define VAR_BUF_DEF1 uint value0_buf1, value1_buf1, value2_buf1, value3_buf1, value4_buf1, value5_buf1, value6_buf1, value7_buf1;
#endif

#if MSG_SIZE_OFFLOAD == 1
#define VAR_BUF_DEF2 uint value0_buf2;
#endif

#if MSG_SIZE_OFFLOAD == 2
#define VAR_BUF_DEF2 uint value0_buf2, value1_buf2;
#endif

#if MSG_SIZE_OFFLOAD == 3
#define VAR_BUF_DEF2 uint value0_buf2, value1_buf2, value2_buf2;
#endif

#if MSG_SIZE_OFFLOAD == 4
#define VAR_BUF_DEF2 uint value0_buf2, value1_buf2, value2_buf2, value3_buf2;
#endif

#if MSG_SIZE_OFFLOAD == 5
#define VAR_BUF_DEF2 uint value0_buf2, value1_buf2, value2_buf2, value3_buf2, value4_buf2;
#endif

#if MSG_SIZE_OFFLOAD == 6
#define VAR_BUF_DEF2 uint value0_buf2, value1_buf2, value2_buf2, value3_buf2, value4_buf2, value5_buf2;
#endif

#if MSG_SIZE_OFFLOAD == 7
#define VAR_BUF_DEF2 uint value0_buf2, value1_buf2, value2_buf2, value3_buf2, value4_buf2, value5_buf2, value6_buf2;
#endif

#if MSG_SIZE_OFFLOAD == 8
#define VAR_BUF_DEF2 uint value0_buf2, value1_buf2, value2_buf2, value3_buf2, value4_buf2, value5_buf2, value6_buf2, value7_buf2;
#endif

#if MSG_SIZE_OFFLOAD == 1
#define VAR_BUF0  value0_buf0
#endif

#if MSG_SIZE_OFFLOAD == 2
#define VAR_BUF0  value0_buf0, value1_buf0
#endif

#if MSG_SIZE_OFFLOAD == 3
#define VAR_BUF0  value0_buf0, value1_buf0, value2_buf0
#endif

#if MSG_SIZE_OFFLOAD == 4
#define VAR_BUF0  value0_buf0, value1_buf0, value2_buf0, value3_buf0
#endif

#if MSG_SIZE_OFFLOAD == 5
#define VAR_BUF0  value0_buf0, value1_buf0, value2_buf0, value3_buf0, value4_buf0
#endif

#if MSG_SIZE_OFFLOAD == 6
#define VAR_BUF0  value0_buf0, value1_buf0, value2_buf0, value3_buf0, value4_buf0, value5_buf0
#endif

#if MSG_SIZE_OFFLOAD == 7
#define VAR_BUF0  value0_buf0, value1_buf0, value2_buf0, value3_buf0, value4_buf0, value5_buf0, value6_buf0
#endif

#if MSG_SIZE_OFFLOAD == 8
#define VAR_BUF0  value0_buf0, value1_buf0, value2_buf0, value3_buf0, value4_buf0, value5_buf0, value6_buf0, value7_buf0
#endif

#if MSG_SIZE_OFFLOAD == 1
#define VAR_BUF1  value0_buf1
#endif

#if MSG_SIZE_OFFLOAD == 2
#define VAR_BUF1  value0_buf1, value1_buf1
#endif

#if MSG_SIZE_OFFLOAD == 3
#define VAR_BUF1  value0_buf1, value1_buf1, value2_buf1
#endif

#if MSG_SIZE_OFFLOAD == 4
#define VAR_BUF1  value0_buf1, value1_buf1, value2_buf1, value3_buf1
#endif

#if MSG_SIZE_OFFLOAD == 5
#define VAR_BUF1  value0_buf1, value1_buf1, value2_buf1, value3_buf1, value4_buf1
#endif

#if MSG_SIZE_OFFLOAD == 6
#define VAR_BUF1  value0_buf1, value1_buf1, value2_buf1, value3_buf1, value4_buf1, value5_buf1
#endif

#if MSG_SIZE_OFFLOAD == 7
#define VAR_BUF1  value0_buf1, value1_buf1, value2_buf1, value3_buf1, value4_buf1, value5_buf1, value6_buf1
#endif

#if MSG_SIZE_OFFLOAD == 8
#define VAR_BUF1  value0_buf1, value1_buf1, value2_buf1, value3_buf1, value4_buf1, value5_buf1, value6_buf1, value7_buf1
#endif

#if MSG_SIZE_OFFLOAD == 1
#define VAR_BUF2  value0_buf2
#endif

#if MSG_SIZE_OFFLOAD == 2
#define VAR_BUF2  value0_buf2, value1_buf2
#endif

#if MSG_SIZE_OFFLOAD == 3
#define VAR_BUF2  value0_buf2, value1_buf2, value2_buf2
#endif

#if MSG_SIZE_OFFLOAD == 4
#define VAR_BUF2  value0_buf2, value1_buf2, value2_buf2, value3_buf2
#endif

#if MSG_SIZE_OFFLOAD == 5
#define VAR_BUF2  value0_buf2, value1_buf2, value2_buf2, value3_buf2, value4_buf2
#endif

#if MSG_SIZE_OFFLOAD == 6
#define VAR_BUF2  value0_buf2, value1_buf2, value2_buf2, value3_buf2, value4_buf2, value5_buf2
#endif

#if MSG_SIZE_OFFLOAD == 7
#define VAR_BUF2  value0_buf2, value1_buf2, value2_buf2, value3_buf2, value4_buf2, value5_buf2, value6_buf2
#endif

#if MSG_SIZE_OFFLOAD == 8
#define VAR_BUF2  value0_buf2, value1_buf2, value2_buf2, value3_buf2, value4_buf2, value5_buf2, value6_buf2, value7_buf2
#endif

__device__ __forceinline__ void get_offload_data(gbc_pack_t gbc_pack,
		uint receiver_id, uint idx, SEND_ARG_DEF_REF) {
#if MSG_SIZE_OFFLOAD > 7
	val7 =gbc_pack.gbc[CHANNEL_OFFLOAD].queue_data[7][receiver_id][idx
	+ get_lane_id()];
#endif
#if MSG_SIZE_OFFLOAD > 6
	val6 =gbc_pack.gbc[CHANNEL_OFFLOAD].queue_data[6][receiver_id][idx
	+ get_lane_id()];
#endif
#if MSG_SIZE_OFFLOAD > 5
	val5 =gbc_pack.gbc[CHANNEL_OFFLOAD].queue_data[5][receiver_id][idx
	+ get_lane_id()];
#endif
#if MSG_SIZE_OFFLOAD > 4
	val4 =gbc_pack.gbc[CHANNEL_OFFLOAD].queue_data[4][receiver_id][idx
	+ get_lane_id()];
#endif
#if MSG_SIZE_OFFLOAD > 3
	val3 =gbc_pack.gbc[CHANNEL_OFFLOAD].queue_data[3][receiver_id][idx
	+ get_lane_id()];
#endif
#if MSG_SIZE_OFFLOAD > 2
	val2 =gbc_pack.gbc[CHANNEL_OFFLOAD].queue_data[2][receiver_id][idx
	+ get_lane_id()];
#endif
#if MSG_SIZE_OFFLOAD > 1
	val1 =gbc_pack.gbc[CHANNEL_OFFLOAD].queue_data[1][receiver_id][idx
	+ get_lane_id()];
#endif
#if MSG_SIZE_OFFLOAD > 0
	val0 =gbc_pack.gbc[CHANNEL_OFFLOAD].queue_data[0][receiver_id][idx
	+ get_lane_id()];
#endif
}

__device__ __forceinline__ void adv_write_gc_data(gbc_t gbc, uint dst,
		uint global_write_ptr, uint work_count, SEND_ARG_DEF) {

	if (get_lane_id() < work_count) {
		uint m_write_idx = (global_write_ptr + get_lane_id())
				& (GC_BUFF_SIZE - 1);
		uint (*queue_data)[NUM_RECEIVER][GC_BUFF_SIZE];
		queue_data = gbc.queue_data;
		uint msg_size = gbc.msg_size;

		switch (msg_size) {
#if MSG_SIZE_MAX > 7
		case 8:
		queue_data[7][dst][m_write_idx] = val7;
		/* no break */
#endif
#if MSG_SIZE_MAX > 6
		case 7:
		queue_data[6][dst][m_write_idx] = val6;
		/* no break */
#endif
#if MSG_SIZE_MAX > 5
		case 6:
		queue_data[5][dst][m_write_idx] = val5;
		/* no break */
#endif
#if MSG_SIZE_MAX > 4
		case 5:
		queue_data[4][dst][m_write_idx] = val4;
		/* no break */
#endif
#if MSG_SIZE_MAX > 3
		case 4:
		queue_data[3][dst][m_write_idx] = val3;
		/* no break */
#endif
#if MSG_SIZE_MAX > 2
		case 3:
		queue_data[2][dst][m_write_idx] = val2;
		/* no break */
#endif
#if MSG_SIZE_MAX > 1
		case 2:
		queue_data[1][dst][m_write_idx] = val1;
		/* no break */
#endif
#if MSG_SIZE_MAX > 0
		case 1:
		queue_data[0][dst][m_write_idx] = val0;
#endif
//   default:
		}
	}
}

__device__ __forceinline__ void adv_read_coal_data(gbc_t gbc, uint dst,
		uint coal_read_ptr, uint work_count,
		uint coal_buf_data[][NUM_RECEIVER][64], SEND_ARG_DEF_REF) {

	if (get_lane_id() < work_count) {
		uint m_read_idx = (coal_read_ptr + get_lane_id()) & (64 - 1);
		uint msg_size = gbc.msg_size;
		switch (msg_size) {
#if MSG_SIZE_MAX > 7
		case 8:
		val7 = coal_buf_data[7][dst][m_read_idx];
		/* no break */
#endif
#if MSG_SIZE_MAX > 6
		case 7:
		val6 = coal_buf_data[6][dst][m_read_idx];
		/* no break */
#endif
#if MSG_SIZE_MAX > 5
		case 6:
		val5 = coal_buf_data[5][dst][m_read_idx];
		/* no break */
#endif
#if MSG_SIZE_MAX > 4
		case 5:
		val4 = coal_buf_data[4][dst][m_read_idx];
		/* no break */
#endif
#if MSG_SIZE_MAX > 3
		case 4:
		val3 = coal_buf_data[3][dst][m_read_idx];
		/* no break */
#endif
#if MSG_SIZE_MAX > 2
		case 3:
		val2 = coal_buf_data[2][dst][m_read_idx];
		/* no break */
#endif
#if MSG_SIZE_MAX > 1
		case 2:
		val1 = coal_buf_data[1][dst][m_read_idx];
		/* no break */
#endif
#if MSG_SIZE_MAX > 0
		case 1:
		val0 = coal_buf_data[0][dst][m_read_idx];
#endif
//		default:
		}
	}
}

__device__ __forceinline__ void adv_transfer_data(gbc_t gbc, uint dst,
		uint coal_read_ptr, uint global_write_ptr, uint work_count,
		uint coal_buf_data[][NUM_RECEIVER][64]) {

	for (int m_offset = get_lane_id(); m_offset < work_count; m_offset += 32) {
		uint m_read_idx = (coal_read_ptr + m_offset) & (64 - 1);
		uint m_write_idx = (global_write_ptr + m_offset) & (GC_BUFF_SIZE - 1);
		uint val;
//read coal buff
//write g buff

		uint (*queue_data)[NUM_RECEIVER][GC_BUFF_SIZE];
		queue_data = gbc.queue_data;
		uint msg_size = gbc.msg_size;

		switch (msg_size) {
#if MSG_SIZE_MAX > 7
		case 8:
		val = coal_buf_data[7][dst][m_read_idx];
		queue_data[7][dst][m_write_idx] = val;
		/* no break */
#endif
#if MSG_SIZE_MAX > 6
		case 7:
		val = coal_buf_data[6][dst][m_read_idx];
		queue_data[6][dst][m_write_idx] = val;
		/* no break */
#endif
#if MSG_SIZE_MAX > 5
		case 6:
		val = coal_buf_data[5][dst][m_read_idx];
		queue_data[5][dst][m_write_idx] = val;
		/* no break */
#endif
#if MSG_SIZE_MAX > 4
		case 5:
		val = coal_buf_data[4][dst][m_read_idx];
		queue_data[4][dst][m_write_idx] = val;
		/* no break */
#endif
#if MSG_SIZE_MAX > 3
		case 4:
		val = coal_buf_data[3][dst][m_read_idx];
		queue_data[3][dst][m_write_idx] = val;
		/* no break */
#endif
#if MSG_SIZE_MAX > 2
		case 3:
		val = coal_buf_data[2][dst][m_read_idx];
		queue_data[2][dst][m_write_idx] = val;
		/* no break */
#endif
#if MSG_SIZE_MAX > 1
		case 2:
		val = coal_buf_data[1][dst][m_read_idx];
		queue_data[1][dst][m_write_idx] = val;
		/* no break */
#endif
#if MSG_SIZE_MAX > 0
		case 1:
		val = coal_buf_data[0][dst][m_read_idx];
		queue_data[0][dst][m_write_idx] = val;
#endif
//		default:
		}
	}

}

__device__ __forceinline__ void adv_write_coal_buf_data(uint dst,
		uint m_write_idx, uint msg_size, SEND_ARG_DEF,
		uint coal_buf_data[][NUM_RECEIVER][64]) {
//write buffer and set bitmask

	switch (msg_size) {
#if MSG_SIZE_MAX > 7
	case 8:
	coal_buf_data[7][dst][m_write_idx] = val7;
	/* no break */
#endif
#if MSG_SIZE_MAX > 6
	case 7:
	coal_buf_data[6][dst][m_write_idx] = val6;
	/* no break */
#endif
#if MSG_SIZE_MAX > 5
	case 6:
	coal_buf_data[5][dst][m_write_idx] = val5;
	/* no break */
#endif
#if MSG_SIZE_MAX > 4
	case 5:
	coal_buf_data[4][dst][m_write_idx] = val4;
	/* no break */
#endif
#if MSG_SIZE_MAX > 3
	case 4:
	coal_buf_data[3][dst][m_write_idx] = val3;
	/* no break */
#endif
#if MSG_SIZE_MAX > 2
	case 3:
	coal_buf_data[2][dst][m_write_idx] = val2;
	/* no break */
#endif
#if MSG_SIZE_MAX > 1
	case 2:
	coal_buf_data[1][dst][m_write_idx] = val1;
	/* no break */
#endif
#if MSG_SIZE_MAX > 0
	case 1:
	coal_buf_data[0][dst][m_write_idx] = val0;
#endif
//	default:
	}
}

__device__ __forceinline__ void base_write_global(gbc_t gbc, uint dst,
		uint write_ptr, SEND_ARG_DEF) {
	uint (*queue_data)[NUM_RECEIVER][GC_BUFF_SIZE];
	queue_data = gbc.queue_data;
	uint msg_size = gbc.msg_size;
	uint write_idx = (write_ptr) & (GC_BUFF_SIZE - 1);
	switch (msg_size) {
#if MSG_SIZE_MAX > 7
	case 8:
	queue_data[7][dst][write_idx] = val7;
	/* no break */
#endif
#if MSG_SIZE_MAX > 6
	case 7:
	queue_data[6][dst][write_idx] = val6;
	/* no break */
#endif
#if MSG_SIZE_MAX > 5
	case 6:
	queue_data[5][dst][write_idx] = val5;
	/* no break */
#endif
#if MSG_SIZE_MAX > 4
	case 5:
	queue_data[4][dst][write_idx] = val4;
	/* no break */
#endif
#if MSG_SIZE_MAX > 3
	case 4:
	queue_data[3][dst][write_idx] = val3;
	/* no break */
#endif
#if MSG_SIZE_MAX > 2
	case 3:
	queue_data[2][dst][write_idx] = val2;
	/* no break */
#endif
#if MSG_SIZE_MAX > 1
	case 2:
	queue_data[1][dst][write_idx] = val1;
	/* no break */
#endif
#if MSG_SIZE_MAX > 0
	case 1:
	queue_data[0][dst][write_idx] = val0;
#endif
//	default:
	}
//set status and reset stuff

}

__device__ void init_base(uint tail_ptr_cpy[NUM_RECEIVER]) {
	if (threadIdx.x == 0) {
		for (int i = 0; i < NUM_RECEIVER; i++) {
			tail_ptr_cpy[i] = 0;
		}
	}
	__syncthreads();
}

__device__ void init_adv(gbc_t gbc, uint channel) {
	if (threadIdx.x == 0) {
		for (int i = 0; i < NUM_RECEIVER; i++) {
			adv_coal_buf_ptrs[channel][0][i] = 0;
			adv_coal_buf_ptrs[channel][1][i] = 0;
			adv_coal_buf_state[channel][i] = 0;
			adv_tail_ptr_cpy[channel][i] = 0;
			adv_reader_states[channel][i] = 0;
		}
		adv_rr_ptr[channel] = 0;
		adv_last_update_time[channel] = get_clock32();
		adv_update_lock[channel] = 0;
		adv_local_exit_counter[channel] =
				gbc.num_sender_per_bk * 32;

	}

	__syncthreads();
}

#endif /* MSG_AUX_H_ */
