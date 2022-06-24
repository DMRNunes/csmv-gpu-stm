/*
 * gb_comm.h
 *
 *  Created on: May 29, 2018
 *      Author: redudie
 */

#ifndef GB_COMM_H_
#define GB_COMM_H_
#include <cuda.h>
#include <inttypes.h>
#include <bitset>
#include <cmath>

/*
 * receiver stuff
 */

///////////////////////////
//gc receiver
__device__ void init_recv(gbc_pack_t gbc_pack) {

	if (threadIdx.x / 32 == 0) {
		for (int channel = 0; channel < NUM_CHANNEL; channel++) {
			uint ini_data = 0;
			uint buff_status = 0;
			if ((get_lane_id() >= (gbc_pack.gbc[channel].receiver_start / 32))
					&& (get_lane_id()
							< (gbc_pack.gbc[channel].receiver_end / 32))) {
				ini_data = NOT_FOUND;
				buff_status = set_bits(buff_status, 0xffffffff, 0,
						gbc_pack.gbc[channel].work_buff_size);
//printf("t%d: init %d\n", thread_id_x(), buff_status);
			}
			for (int i = 0; i < gbc_pack.gbc[channel].work_buff_size; i++) {
				work_buffer[channel][i][get_lane_id()] = ini_data;
			}
			work_buffer_status[channel][get_lane_id()] = buff_status;

			if (get_lane_id() == 0) {
				for (int i = 0; i < NUM_RESET_BUF; i++) {
					reset_accu[channel][i] = 0;
					reset_target[channel][i] = 0;
				}
				reset_head[channel] = 0;
				reset_tail[channel] = 0;
				local_tail_ptr[channel] = 0;

				leader_terminate[channel] = false;
				leader_exit_check[channel] = 0;
				leader_head_ptr[channel] = 0;
				leader_new_head_ptr[channel] = 0;
				leader_assign_reset_buf[channel] = 0;
			}

		}
	}
	__syncthreads();
}

__device__ bool leader_exit_channel(gbc_t target_gbc, uint channel) {

	for (int buf_id = 0; buf_id < target_gbc.work_buff_size; buf_id++) {
		uint still_working = 0;
		if ((get_lane_id() >= (target_gbc.receiver_start / 32))
				&& (get_lane_id() < (target_gbc.receiver_end / 32))) {
			uint buff = work_buffer[channel][buf_id][get_lane_id()];
			still_working = (buff != NOT_FOUND) ? 1 : 0;
		}
		if (vote_ballot(still_working) != 0) {
			return false;
		}
	}
	//now all workers are done

	for (int buf_id = 0; buf_id < target_gbc.work_buff_size; buf_id++) {
		if ((get_lane_id() >= (target_gbc.receiver_start / 32))
				&& (get_lane_id() < (target_gbc.receiver_end / 32))) {
			work_buffer[channel][buf_id][get_lane_id()] = GC_TERM;
		}
	}
	return true;
}

__device__ void gc_receiver_leader(gbc_pack_t gbc_pack) {

	//check if this warp can be any channel's leader
	bool any_leader = false;
	int channel_exit_counter = 0;
	int channel_begin = 1000;
	int channel_end = -1;
	for (int channel = 0; channel < NUM_CHANNEL; channel++) {
		if ((threadIdx.x / 32) == gbc_pack.gbc[channel].leader_warp_id) {
			any_leader = true;
			channel_exit_counter++;
			channel_begin = min(channel, channel_begin);
			channel_end = max(channel, channel_end);
		}
	}

	if (!any_leader) {
		return;
	}

#if PRINT_DEBUG  == 0
	int print_counter = 0;
#endif

	while (channel_exit_counter != 0) {

		for (int channel = channel_begin; channel <= channel_end; channel++) {
			if (!leader_terminate[channel]) {
				//exit check
				//end condition

				uint head_ptr = leader_head_ptr[channel];
				uint new_head_ptr = leader_new_head_ptr[channel];
				uint reset_head_ptr = reset_head[channel];
				uint reset_tail_ptr = reset_tail[channel];

				if ((head_ptr == new_head_ptr)
						&& ((reset_head_ptr - reset_tail_ptr) < NUM_RESET_BUF)) {

					//check for available work passing the head ptr
					uint head_word = head_ptr >> 5; //word id 					head_word will be used to check the mask, hence divided by 32
					//memory transaction aligned to cache line
					//translate ptr to idx for actual memory access
					uint m_word = (head_word + get_lane_id())					// add lane id to the head_word to find the corresponding word in the mask
							& (GC_NUM_MASK - 1);
					uint m_bit_mask =
							gbc_pack.gbc[channel].ready_bitmask[block_id_x()][m_word];	//extract the mask using the previous word
//printf("\tS%d: mask %x word %d\n", get_lane_id(), m_bit_mask, m_word);					
					//inverse to find 0
					m_bit_mask = ~m_bit_mask;
					//filter the first lane
					if (get_lane_id() == 0) {
						m_bit_mask = set_bits(m_bit_mask, 0, 0,
								(head_ptr & 0x1f));
					}

					//find the first 0
					int last_bit_idx = NOT_FOUND;
					last_bit_idx = find_nth_bit(m_bit_mask, 0, 1);		//find the first 0 bit in the lane's mask

					int non_consec = (last_bit_idx == NOT_FOUND) ? 0 : 1;//whether I have a non-consec bitmask

					//do a vote and find the first non-consec lane
					uint vote_result = vote_ballot(non_consec);
					uint first_lane = find_nth_bit(vote_result, 0, 1);
					uint src_last_bit_idx = shuffle_idx(last_bit_idx,	//sends the value of last_bit_idx from first lane that has observed a 0 bit in the mask
							first_lane);
					if (first_lane == NOT_FOUND) {
						//all consec for the current cache line
						//set head to the next cacheline
						new_head_ptr = (head_word + 32) << 5;
					} else {
						new_head_ptr = ((head_word + first_lane) << 5);
								//+ src_last_bit_idx;
						//if ((get_lane_id() == 0)&& ((src_last_bit_idx)!=0))
						//	printf("S%d: should not be here. mask %x nhp %d hw %d fl %d lb %d\n", get_lane_id(), m_bit_mask, new_head_ptr, head_word, first_lane, src_last_bit_idx);
					}

					if (get_lane_id() == 0) {
						uint head_reset_buf = reset_head_ptr % NUM_RESET_BUF;
						leader_new_head_ptr[channel] = new_head_ptr;
						uint cur_reset_count =
								reset_target[channel][head_reset_buf];
						cur_reset_count += (new_head_ptr - head_ptr);
						reset_target[channel][head_reset_buf] = cur_reset_count;
						leader_assign_reset_buf[channel] = head_reset_buf;
						if (cur_reset_count >= RESET_THD) {
							reset_head[channel] = (reset_head_ptr + 1);
						}
					}
				}

#if PRINT_DEBUG  ==  1
				if ((get_lane_id() == 0)&& ((new_head_ptr&0x1f)!=0)){// && print_counter == 0) {
					printf(
							"channel %d recv id %d head ptr %u new_head_ptr %u reserve ptr %u global tail %u reset head %u reset tail %u\n",
							channel, block_id_x(), leader_head_ptr[channel],
							leader_new_head_ptr[channel],
							gbc_pack.gbc[channel].write_ptr[block_id_x()],
							gbc_pack.gbc[channel].global_tail_ptr[block_id_x()]),
							reset_head_ptr,
							reset_tail_ptr;

				}
#endif

				//assign work to workers
				if ((head_ptr <= new_head_ptr) )  {
					//get status mask from the work buffer for each lane
					uint status_mask =
							work_buffer_status[channel][get_lane_id()];
					//using the mask, find the buffer with the lowest id that is available
					uint available_buffer = find_nth_bit(status_mask, 0, 1);	//find the first 0 bit in the status_mask

					//if no worker buffer is currently available
					int work_ballot = 1;
					if (available_buffer == NOT_FOUND) {
						work_ballot = 0;
					}
					//perform ballot over which lanes have work buffers free or not
					uint vote_result = vote_ballot(work_ballot);
					//count the number of lanes that have been marked as having available buffer from my lane up to the last lane
					uint m_assignment = count_bit(							//count bits: Count the number of 1 bits
							set_bits(vote_result, 0, get_lane_id(), 32));	//Set bits: Align and insert a bit field from vote_result into 0. 
																			//lane id is the starting bit position for insert, and 32 is the bit field length in bits.

					//the last lane in the group
					bool last_assignment = (set_bits(vote_result, 0, 0,		//set bits: puts vote_result in 0. Starting from idx 0 up to lane+1
							get_lane_id() + 1) == 0);						//compares it with 0 and stores the result in last_assigment

					uint start_size = min((new_head_ptr - head_ptr),		//On almost all cases this value will be 32
							32 - (head_ptr & 0x1f));						//32 - last 5 bits from head_ptr = 32 - head_ptr%32

					uint m_end_ptr = (head_ptr + start_size)				//m_end_ptr = head_ptr+start_size + (n of lanes available)*32
							+ (m_assignment << 5);
					uint assign_ptr = m_end_ptr - 32;						
					uint assign_size = 32;

					if (m_end_ptr > new_head_ptr) {							//if m_end_ptr surpasses the new_head_ptr aka if there are more than enough lanes to which to write messages
						assign_size = new_head_ptr & 0x1f;					//assign size = new_head_ptr % 32
					}
					//if no lanes are available
					if (m_assignment == 0) {		
						assign_ptr = head_ptr;								
						assign_size = start_size;
					}
#if PRINT_DEBUG  == 0
	printf("\tS%d: head %u new_head %u m_end_ptr %u assign_ptr %u assign_size %u m_assignment %u vote_result %x last_assignment %d\n",
						get_lane_id(), head_ptr, new_head_ptr, m_end_ptr, 
						assign_ptr, assign_size, m_assignment, vote_result, last_assignment);
#endif
					if ((work_ballot == 1) ) {			//only if this lane has available buffers do we enter this if
						//write to buffer
						//write to reg warps' buffer
						//translate point to buffer idx
						if (assign_ptr < new_head_ptr) {					
							uint idx = (assign_ptr) & (GC_BUFF_SIZE - 1);
							uint t = set_bits(idx, assign_size,
									AB_SIZE_START_BIT, AB_SIZE_NUM_BIT);
							t = set_bits(t, leader_assign_reset_buf[channel],
									AB_RESET_START_BIT, AB_RESET_NUM_BIT);
#if PRINT_DEBUG  == 1
if(assign_size != 32)
	printf("\tS%d: t %x idx %d size %d head_ptr %d new_head_ptr %d\n", get_lane_id(), t, assign_ptr, assign_size, head_ptr, new_head_ptr);
#endif
							work_buffer[channel][available_buffer][get_lane_id()] =
									t;
							//reset bit
							atomicAnd(
									&(work_buffer_status[channel][get_lane_id()]),
									~(1 << available_buffer));
						}
						//reset condition
						if (last_assignment) {
							//then we must exceed the new_head_ptr, reset
							if (m_end_ptr >= new_head_ptr) {
								leader_head_ptr[channel] =
										leader_new_head_ptr[channel];
							} else {
								//update head ptr
								leader_head_ptr[channel] = assign_ptr
										+ assign_size;
							}
							if((leader_head_ptr[channel] & 0x1F) != 0)
								printf("\tS%d: head %u new_head %u m_end_ptr %u assign_ptr %u assign_size %u m_assignment %u vote_result %x last_assignment %d\n",
										get_lane_id(), head_ptr, new_head_ptr, m_end_ptr, 
										assign_ptr, assign_size, m_assignment, vote_result, last_assignment);
						}

					}
				}

				//reset global
				uint tail_reset_buf = reset_tail_ptr % NUM_RESET_BUF;
				uint target_count = reset_target[channel][tail_reset_buf];
				uint accu_count = reset_accu[channel][tail_reset_buf];
				if ((target_count == accu_count)
						&& (target_count >= RESET_THD)) {
					uint tail_ptr = local_tail_ptr[channel];

					//reset bitmask
					uint start_idx = tail_ptr;
					uint end_idx = start_idx + target_count;
					uint start_word = start_idx >> 5;
					uint end_word = end_idx >> 5;

					for (uint m_word = start_word + get_lane_id();
							m_word <= end_word; m_word += 32) {
						//GC_NUM_MASK
						uint mask = 0;
						if (m_word == start_word) {
							//num of msg from boundary
							mask = set_bits(0, 0xffffffff, 0, start_idx & 0x1f);
						} else if (m_word == end_word) {
							mask = set_bits(0xffffffff, 0, 0, end_idx & 0x1f);
						}
						atomicAnd(
								&(gbc_pack.gbc[channel].ready_bitmask[block_id_x()][m_word
										& (GC_NUM_MASK - 1)]), mask);
					}
					if (get_lane_id() == 0) {
						//reset local
						local_tail_ptr[channel] = tail_ptr + target_count;
						reset_target[channel][tail_reset_buf] = 0;
						reset_accu[channel][tail_reset_buf] = 0;
						reset_tail[channel] = (reset_tail_ptr + 1);
						//flush
						__threadfence();
						atomicAdd(
								&(gbc_pack.gbc[channel].global_tail_ptr[block_id_x()]),
								target_count);
					}
				}
				//when clients end they change the value of exit_check_counter
				//this server knows when to exit when all the clients are finished
				//exit condition check
				uint exit_check_counter = leader_exit_check[channel];
				if (exit_check_counter == 256) {
					if ((*(gbc_pack.gbc[channel].exit_counter_global)) == 0) {
						if (get_lane_id() == 0) {
							leader_exit_check[channel] = 512;
						}
					} else {
						if (get_lane_id() == 0) {
							leader_exit_check[channel] = 0;
						}
					}
				} else if (exit_check_counter == 512) {
					//the point is that we must check bitmask array after G count reaches 0
					if (leader_head_ptr[channel]
							== gbc_pack.gbc[channel].write_ptr[block_id_x()]) {
						if (leader_exit_channel(gbc_pack.gbc[channel],
								channel)) {
							leader_terminate[channel] = true;
							channel_exit_counter--;
						}
					}
				} else {
					if (get_lane_id() == 0) {
						leader_exit_check[channel]++;
					}
				}

			}

		}

		/////////////////////////////////////////////////
		/////////////////////////////////////////////////
#if PRINT_DEBUG  ==  0
		print_counter++;
		//if (print_counter == 1) {
		if (print_counter == 1024 * 8) {
			print_counter = 0;
		}

#endif
		/////////////////////////////////////////////////
		/////////////////////////////////////////////////

	}

	//exit
	asm volatile("exit;");

}

__device__ void gc_receiver_leader_worker(gbc_t gbc, void func(uint data)) {
	if ((threadIdx.x / 32) == gbc.leader_worker_warp_id) {
#if 0
		uint total_req = 0;
		uint total_iter = 0;
#endif
		uint head_ptr = 0;
		//power of 2
#if PRINT_DEBUG  == 0
		int print_counter = 0;
#endif
		//break_pt();
		//loop
		bool terminate = false;
		bool pre_terminate = false;
		uint exit_check = 0;
		uint reset_agg = 0;

		while (!terminate) {
			uint new_head_ptr = head_ptr;

			//check for available work passing the head ptr
			uint head_word = head_ptr >> 5; //word id
			//memory transaction aligned to cache line
			//translate ptr to idx for actual memory access
			uint m_word = (head_word + get_lane_id()) & (GC_NUM_MASK - 1);
			uint m_bit_mask = gbc.ready_bitmask[block_id_x()][m_word];

			//inverse to find 0
			m_bit_mask = ~m_bit_mask;
			//filter the first lane
			if (get_lane_id() == 0) {
				m_bit_mask = set_bits(m_bit_mask, 0, 0, (head_ptr & 0x1f));
			}

			//find the first 0
			int last_bit_idx = NOT_FOUND;
			last_bit_idx = find_nth_bit(m_bit_mask, 0, 1);
			int non_consec = (last_bit_idx == NOT_FOUND) ? 0 : 1;//whether I have a non-consec bitmask

			//do a vote and find the first non-consec lane
			uint vote_result = vote_ballot(non_consec);
			uint first_lane = find_nth_bit(vote_result, 0, 1);
			uint src_last_bit_idx = shuffle_idx(last_bit_idx, first_lane);
			if (first_lane == NOT_FOUND) {
				//all consec for the current cache line
				//set head to the next cacheline
				new_head_ptr = (head_word + 32) << 5;
			} else {
				new_head_ptr = ((head_word + first_lane) << 5)
						+ src_last_bit_idx;
			}

			//unlock routine
			uint work_count = new_head_ptr - head_ptr;

			if (work_count > 0) {
#if 0
				total_iter++;
				total_req += work_count;
#endif
				for (int ptr = head_ptr + get_lane_id(); ptr < new_head_ptr;
						ptr += 32) {
					uint idx = (ptr) & (GC_BUFF_SIZE - 1);
					uint data = gbc.queue_data[0][block_id_x()][idx];
					func(data);
				}
				reset_agg += work_count;

				if (reset_agg >= RESET_THD) {
					uint start_idx = new_head_ptr - reset_agg;
					uint end_idx = new_head_ptr;
					uint start_word = start_idx >> 5;
					uint end_word = end_idx >> 5;

					for (uint m_word = start_word + get_lane_id();
							m_word <= end_word; m_word += 32) {
						//GC_NUM_MASK
						uint mask = 0;
						if (m_word == start_word) {
							//num of msg from boundary
							mask = set_bits(0, 0xffffffff, 0, start_idx & 0x1f);
						} else if (m_word == end_word) {
							mask = set_bits(0xffffffff, 0, 0, end_idx & 0x1f);
						}
						atomicAnd(
								&(gbc.ready_bitmask[block_id_x()][m_word
										& (GC_NUM_MASK - 1)]), mask);
					}

					if (get_lane_id() == 0) {
						//reset global tail
						atomicAdd(&(gbc.global_tail_ptr[block_id_x()]),
								reset_agg);
					}
					reset_agg = 0;
				}

			}

			//end condition
			if (exit_check == 256) {

				if (pre_terminate) {
					if (head_ptr == new_head_ptr) {
						terminate = true;
					}
				} else if ((*(gbc.exit_counter_global)) == 0) {
					pre_terminate = true;
				}
				exit_check = 0;
			}
			exit_check++;

#if PRINT_DEBUG  == 0
			if (get_lane_id() == 0 && print_counter == 1) {
				printf(
						"leader worker recv id %d warp id %d head ptr %u new_head_ptr %u reserve ptr %u global tail %u\n",
						block_id_x(), thread_id_x() / 32, head_ptr,
						new_head_ptr, gbc.write_ptr[block_id_x()],
						gbc.global_tail_ptr[block_id_x()]);

			}
			if (print_counter == 1) {
				print_counter = 0;
			}

			print_counter++;
#endif

			head_ptr = new_head_ptr;

		}

#if 0
		if (get_lane_id() == 0) {
			printf("bid %u, wid %u, average %f\n", block_id_x(),
					threadIdx.x / 32, (float) total_req / (float) total_iter);
		}
#endif
		asm volatile("exit;");
	}

}

__device__ __forceinline__ unsigned gc_get_work_buff(uint channel, uint warp_id,
		uint buf_id) {
	return work_buffer[channel][buf_id][warp_id];
}

__device__ __forceinline__ void gc_reset_work_buff(uint channel, uint warp_id,
		uint buf_id) {
	if (get_lane_id() == 0) {
		work_buffer[channel][buf_id][warp_id] = NOT_FOUND;
		__threadfence_block();
		atomicOr(&(work_buffer_status[channel][warp_id]), (1 << buf_id));
	}
}

__device__ __forceinline__ void gc_receiver_reset_mp_status(gbc_pack_t gbc_pack,
		uint channel, uint warp_id, uint buf_id) {

	uint status = work_buffer[channel][buf_id][warp_id];
	uint size = extract_bits(status, AB_SIZE_START_BIT, AB_SIZE_NUM_BIT);
	uint m_reset_buf_id = extract_bits(status, AB_RESET_START_BIT,
			AB_RESET_NUM_BIT);

	if (get_lane_id() == 0) {
		atomicAdd(&(reset_accu[channel][m_reset_buf_id]), size);
	}
}

/*
 * sender stuff
 */

///////////////////////////
//gc sender
__device__ void base_exit(gbc_t gbc) {
	__threadfence();
	if (get_lane_id() == 0) {
		atomicAdd(gbc.exit_counter_global, -32);
	}
}

__device__ __forceinline__ bool base_check_avail(uint dst, uint write_ptr,
		uint* tail_ptr_cpy) {
//make sure there is enough buffer space
	uint local_tail = tail_ptr_cpy[dst];
	if (((write_ptr + 32) - local_tail) < GC_BUFF_SIZE) {
		return true;
	}

	return false;
}

__device__ void base_set_global(uint dst, uint write_ptr,
		uint ready_bitmask[][GC_NUM_MASK]) {
//set the status bit
//uint tid = threadIdx.x + blockIdx.x * blockDim.x;

	uint write_idx = (write_ptr) & (GC_BUFF_SIZE - 1);
	int word_id = write_idx >> 5;
	int bit_id = write_idx & 0x1f;
	uint mask = set_bits(0, 0xffffffff, bit_id, 1);
	atomicOr(&(ready_bitmask[dst][word_id]), mask);
//if(get_lane_id()==0)printf("C%d: mask %x %x word %u\n", tid/32, mask, ready_bitmask[dst][word_id], word_id);
}

__device__ __forceinline__ void base_update_tail(gbc_t gbc,
		uint tail_ptr_cpy[NUM_RECEIVER]) {
//update local counter when needed
	if (get_lane_id() < NUM_RECEIVER) {
		tail_ptr_cpy[get_lane_id()] = gbc.global_tail_ptr[get_lane_id()];
	}
}

/////////////////////////////////////////////////////////
__device__ __forceinline__ bool base_send(gbc_t gbc, uint dst, uint valid_msg,
		uint& saved_write_ptr, uint tail_ptr_cpy[NUM_RECEIVER], SEND_ARG_DEF) {
	bool success = true;
	uint update_global_tail = 0;
	if (valid_msg) {
		//reserve global idx
		
		
		
		//if(saved_write_ptr >= GC_BUFF_SIZE)
		//	saved_write_ptr = atomicCAS(&(gbc.write_ptr[dst]), GC_BUFF_SIZE, 0);

		uint write_ptr = saved_write_ptr;
		if (base_check_avail(dst, write_ptr, tail_ptr_cpy)) {
			//there is enough space
			base_write_global(gbc, dst, write_ptr, SEND_ARG);
			__threadfence();
			//set global status
			base_set_global(dst, write_ptr, gbc.ready_bitmask);
#if 0
			if (gbc.leader_warp_id == 0 && gbc.msg_size == 2) {
				printf(
						"send msg lock b:%u,l:%u, from b:%u,tid:%u,tk:%u, write ptr %u\n",
						dst, extract_bits(val0, 0, 14),
						extract_bits(val0, 24, 5), extract_bits(val0, 14, 10),
						extract_bits(val1, 0, 16), write_ptr);
			}
#endif
		} else {
			update_global_tail = 1;
			success = false;
		}
	}
	if (vote_ballot(update_global_tail) != 0) {
		base_update_tail(gbc, tail_ptr_cpy);
	}
	return success;

}
/////////////////////////////////////////////////////////
__device__ __forceinline__ void base_update_tail_individual(gbc_t gbc, uint dst,
		uint tail_ptr_cpy[NUM_RECEIVER]) {
//update local counter when needed
	tail_ptr_cpy[dst] = gbc.global_tail_ptr[dst];
}

__device__ __forceinline__ void base_send_individual(gbc_t gbc, uint dst,
		uint tail_ptr_cpy[NUM_RECEIVER], SEND_ARG_DEF) {

	bool success = false;
	uint write_ptr = atomicAdd(&(gbc.write_ptr[dst]), 1);
	do {
		if (base_check_avail(dst, write_ptr, tail_ptr_cpy)) {
			//there is enough space
			base_write_global(gbc, dst, write_ptr, SEND_ARG);
			__threadfence();
			//set global status
			base_set_global(dst, write_ptr, gbc.ready_bitmask);
			success = true;
#if 0

			if (gbc.leader_warp_id == 7 && gbc.msg_size == 2) {

				printf(
						"inv send msg lock b:%u,l:%u, from b:%u,tid:%u,tk:%u, write ptr %u, channel %u\n",
						dst, extract_bits(val0, 0, 14),
						extract_bits(val0, 24, 5), extract_bits(val0, 14, 10),
						extract_bits(val1, 0, 16), write_ptr, gbc.channel_id);
			}
#endif
		} else {
			base_update_tail_individual(gbc, dst, tail_ptr_cpy);
		}

	} while (!success);
}
//////////////////////////////////////
//gbc adv sender
#define POL_MASK 0x80000000
#define ADV_LOCK_MASK 0x40000000
#define RESV_BITS 26
#define WRT_BITS 6
#define RESV_MASK 0x3ffffff

__device__ __forceinline__ void adv_try_resv_coal_buf(uint channel_id, uint dst,
		uint& ret_polarity, uint& ret_ptr) {

	//volatile uint polarity = coal_buf_polarity[dst];

	volatile uint polarity = extract_bits(adv_coal_buf_state[channel_id][dst],
			31, 1);
	//try 1st
	uint ptr = (atomicAdd(&(adv_coal_buf_ptrs[channel_id][polarity][dst]), 1))
			& RESV_MASK;
	if (ptr > 31) {
		//try 2nd
		polarity ^= 1;
		ptr = (atomicAdd(&(adv_coal_buf_ptrs[channel_id][polarity][dst]), 1))
				& RESV_MASK;
	}
	if (ptr == 31) {
		//flip
		atomicXor(&(adv_coal_buf_state[channel_id][dst]), POL_MASK);
	}
	ret_polarity = polarity;
	ret_ptr = ptr;
}

__device__ __forceinline__ unsigned adv_check_coal_buf(uint dst, uint ptr,
		bool& check_next, uint status) {

	uint write_ptr = extract_bits(status, RESV_BITS,
	WRT_BITS);
	//__threadfence_block();
	//the memfence here is to ensure memory reads ordering
	//for the two variables` accesses
	uint resv_ptr = extract_bits(status, 0, RESV_BITS);
	//note that write_ptr is read before resv_ptr
	//when write_ptr == resv_ptr, it is impossible for
	//the region < write_ptr to be invalid
	if ((write_ptr == 32) || (write_ptr == resv_ptr)) {
		if (write_ptr == 32) {
			check_next = true;
		}
		return (write_ptr - ptr);
	}
	//else do nothing,
	//it means that there are still undergoing writes
	//wait it to coalesce more
	return 0;
}

__device__ __forceinline__ unsigned adv_find_available_work(uint channel_id,
		uint dst, uint m_read_ptr) {
	uint m_available_work = 0;
	uint polarity = extract_bits(m_read_ptr, 5, 1);
	uint ptr = extract_bits(m_read_ptr, 0, 5);
	bool check_next = false;
	//check 1st buf
	uint ptr_pack = adv_coal_buf_ptrs[channel_id][polarity][dst];
	m_available_work += adv_check_coal_buf(dst, ptr, check_next, ptr_pack);

	if (check_next) {
		polarity ^= 1;
		ptr = 0;
		ptr_pack = adv_coal_buf_ptrs[channel_id][polarity][dst];
		m_available_work += adv_check_coal_buf(dst, ptr, check_next, ptr_pack);
	}
	return m_available_work;
}

__device__ __forceinline__ void adv_reset_local(uint channel_id, uint dst,
		uint read_ptr, uint work_count) {
	if (get_lane_id() == 0) {
		//reset local status
		{
			uint polarity = extract_bits(read_ptr, 5, 1);
			uint ptr = extract_bits(read_ptr, 0, 5);
			uint new_count = work_count + ptr;
			if (new_count >= 32) {
				atomicExch(&(adv_coal_buf_ptrs[channel_id][polarity][dst]), 0);
				bool set_second = (new_count == 64);
				if (set_second) {
					atomicExch(
							&(adv_coal_buf_ptrs[channel_id][polarity ^ 1][dst]),
							0);
				}
			}
			adv_reader_states[channel_id][dst] = (read_ptr + work_count)
					& (64 - 1);
			//zero-ed the bits
		}
	}
}

__device__ __forceinline__ void adv_set_global(uint dst, uint g_write_ptr,
		uint work_count, gbc_t gbc) {
	//set global status
	uint m_word = (g_write_ptr >> 5) + get_lane_id();
	uint first_bit = g_write_ptr & 0x1f;
	uint start_bit = first_bit;
	int bit_count = work_count;
	if (get_lane_id() != 0) {
		start_bit = 0;
		bit_count -= ((32 * get_lane_id()) - first_bit);
	}

	if (bit_count > 0) {
		uint idx = m_word & (GC_NUM_MASK - 1);
		uint mask = set_bits(0, 0xffffffff, start_bit, (uint) bit_count);
		atomicOr(&(gbc.ready_bitmask[dst][idx]), mask);
	}
}

__device__ __forceinline__ int adv_get_free_size(uint channel_id, uint dst,
		uint m_write_ptr) {

	int m_local_tail = (int) adv_tail_ptr_cpy[channel_id][dst];
	return (GC_BUFF_SIZE - 32) - ((int) m_write_ptr - m_local_tail);
}

__device__ __forceinline__ void adv_update_tail(gbc_t gbc, uint channel_id,
		int free_size) {

	if (free_size < (GC_BUFF_SIZE / 4)) {
		bool lane_0 = (get_lane_id() == 0);
		uint old_val;
		if (lane_0) {
			old_val = atomicCAS(&(adv_update_lock[channel_id]), 0, 1);
		}
		old_val = shuffle_idx(old_val, 0);
		if (old_val == 0) {
			unsigned cur_time = get_clock32();
			unsigned last_time = adv_last_update_time[channel_id];
			if ((cur_time < last_time)
					|| ((cur_time - last_time) > TAIL_UPDATE_THD)) {
				if (get_lane_id() < NUM_RECEIVER) {
					adv_tail_ptr_cpy[channel_id][get_lane_id()] =
							gbc.global_tail_ptr[get_lane_id()];
				}
				if (lane_0) {
					adv_last_update_time[channel_id] = cur_time;
				}
			}
			if (lane_0) {
				atomicExch(&(adv_update_lock[channel_id]), 0);
			}
		}
	}
}

__device__ __forceinline__ void adv_global_action(gbc_t gbc, uint channel_id,
		uint coal_buf_data[][NUM_RECEIVER][64]) {

	bool is_lane0 = (get_lane_id() == 0);
	uint dst;
	uint old_lock_val;
	if (is_lane0) {
		//find a dst in RR fashion within the TB
		dst = atomicAdd(&(adv_rr_ptr[channel_id]), 1) % NUM_RECEIVER;
		//lock the dst
		old_lock_val = atomicOr(&(adv_coal_buf_state[channel_id][dst]),
		ADV_LOCK_MASK) & ADV_LOCK_MASK;
	}
	dst = shuffle_idx(dst, 0);
	old_lock_val = shuffle_idx(old_lock_val, 0);

	if (old_lock_val == 0) {
		int g_free_size = GC_BUFF_SIZE;
		//check the coal buffer for work
		uint state = adv_reader_states[channel_id][dst];
		uint read_ptr = extract_bits(state, 0, 7);
		uint work_count = extract_bits(state, 7, 7);
		uint g_write_ptr;
		if (work_count != 0) {
			//there are previously unsent work, do those first
			g_write_ptr = adv_saved_write_ptr[channel_id][dst];
		} else {
			uint m_available_work = adv_find_available_work(channel_id, dst,
					read_ptr);
			work_count = m_available_work;
			//reserve a range of write location
			if (is_lane0 && work_count > 0) {
				g_write_ptr = atomicAdd(&(gbc.write_ptr[dst]), work_count);
			}
			g_write_ptr = shuffle_idx(g_write_ptr, 0);
		}
		if (work_count > 0) {
			g_free_size = adv_get_free_size(channel_id, dst, g_write_ptr);
			if (g_free_size >= (int) work_count) {
				//enough space
				//now write the coal buf content to global buffer
				adv_transfer_data(gbc, dst, read_ptr, g_write_ptr, work_count,
						coal_buf_data);
				adv_reset_local(channel_id, dst, read_ptr, work_count);
				__threadfence();
				adv_set_global(dst, g_write_ptr, work_count, gbc);

			} else if (get_lane_id() == 0) {
				//not enough space
				//save it for future
				adv_saved_write_ptr[channel_id][dst] = g_write_ptr;
				uint save_state = set_bits(read_ptr, work_count, 7, 7);
				adv_reader_states[channel_id][dst] = save_state;
			}
		}
		//unlock
		if (is_lane0) {
			__threadfence_block();
			atomicAnd(&(adv_coal_buf_state[channel_id][dst]), ~ADV_LOCK_MASK);
		}
		//update global tail
		adv_update_tail(gbc, channel_id, g_free_size);
	}

}

__device__ __forceinline__ bool adv_send_msg(gbc_t gbc, uint channel_id,
		uint dst, uint valid_msg, uint coal_buf_data[][NUM_RECEIVER][64],
		SEND_ARG_DEF) {

	bool success = true;
	if (valid_msg) {
		uint ptr;
		uint polarity;
		//write lock buffer first
		adv_try_resv_coal_buf(channel_id, dst, polarity, ptr);
		if (ptr < 32) {
			uint m_write_idx = polarity * 32 + ptr;
			adv_write_coal_buf_data(dst, m_write_idx, gbc.msg_size, SEND_ARG,
					coal_buf_data);
			__threadfence_block();
			//set write_ptr, to indicated 1 msg write
			atomicAdd(&(adv_coal_buf_ptrs[channel_id][polarity][dst]),
					1 << RESV_BITS);

		} else {
			success = false;
		}
	}
//do global stuff
	adv_global_action(gbc, channel_id, coal_buf_data);

	return success;
}

__device__ void adv_exit(gbc_t gbc, uint channel_id,
		uint coal_buf_data[][NUM_RECEIVER][64]) {

	__threadfence_block();
	uint cur_local_count;
	if (get_lane_id() == 0) {
		cur_local_count = atomicAdd(&(adv_local_exit_counter[channel_id]), -32);
	}
	cur_local_count = shuffle_idx(cur_local_count, 0);
	if (cur_local_count == 32) {
		//flush all msg
		bool terminate = false;
		while (!terminate) {

			adv_global_action(gbc, channel_id, coal_buf_data);

			//check exit condition
			{
				uint dst = get_lane_id();
				uint available_work = 0;
				if (dst < NUM_RECEIVER) {
					uint m_read_ptr = extract_bits(
							adv_reader_states[channel_id][dst], 0, 7);
					available_work = adv_find_available_work(channel_id, dst,
							m_read_ptr);
				}
				uint ballot = (available_work > 0) ? 1 : 0;
				uint vote_result = vote_ballot(ballot);
				if (vote_result == 0) {
					terminate = true;
				} else {
				}
			}
		}
		__threadfence();
		if (get_lane_id() == 0) {
			atomicSub(gbc.exit_counter_global, gbc.num_sender_per_bk * 32);
		}
	}

}

#endif /* GB_COMM_H_ */
