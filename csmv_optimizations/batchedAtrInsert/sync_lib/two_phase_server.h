/*
 * two_phase_server.h
 *
 *  Created on: Jan 3, 2019
 *      Author: redudie
 */

#ifndef TWO_PHASE_SERVER_H_
#define TWO_PHASE_SERVER_H_

#define GRAB_VERSION BASE_SEND
#define UNLOCK_VERSION BASE_SEND
#define REPLY_VERSION BASE_SEND

#define LOCK_MASK 0x80000000
__shared__ uint total_req[NUM_CHANNEL];
__shared__ uint total_iter[NUM_CHANNEL];
__shared__ uint _grab_tail_ptr_cpy[NUM_RECEIVER];
__shared__ uint _reply_tail_ptr_cpy[NUM_RECEIVER];
__shared__ uint _unlock_tail_ptr_cpy[NUM_RECEIVER];

///////
__shared__ uint _grab_coal_buf_data[MSG_SIZE_GRAB][NUM_RECEIVER][64];
__shared__ uint _reply_coal_buf_data[MSG_SIZE_REPLY][NUM_RECEIVER][64];
__shared__ uint _unlock_coal_buf_data[MSG_SIZE_UNLOCK][NUM_RECEIVER][64];
///////

__shared__ uint lock_struct[NUM_LOCK];
__shared__ uint p1_signal[1024 / 32];

__device__ void init_server() {
	for (int i = threadIdx.x; i < NUM_LOCK; i += blockDim.x) {
		lock_struct[i] = 0;
	}

	for (int i = threadIdx.x; i < 1024 / 32; i += blockDim.x) {
		p1_signal[i] = 0;
	}

	__syncthreads();
}

__device__ void gc_unlock_func(uint data) {
	uint lock_id = data;
	//the bit is 0
	atomicAnd(&(lock_struct[lock_id]), ~LOCK_MASK);
}

__device__ void gc_reply_func(uint data) {
	uint target_tid = extract_bits(data, 14, 10);
	uint warp_id = target_tid >> 5;
	uint bitmask = 1 << (target_tid & 0x1f);
	atomicOr(&(p1_signal[warp_id]), bitmask);
}

//two phase server
void create_gbc(gbc_pack_t & gbc_pack, int total_client_bk,
		int client_block_size, int server_block_size) {

	///////////////////////////////////////////////
	//gbc init
#if SHARE_GRAB == 0
#define NUM_LEADER 4
#else
#define NUM_LEADER 3
#endif

#define PHASE_RATIO 2
	int num_receiver_per_bk = server_block_size / 32;
	num_receiver_per_bk -= NUM_LEADER;

	uint t = (num_receiver_per_bk);
	uint num_grab_recv_warp = NUM_GRAB_WARP;
	uint num_offload_recv_warp = num_receiver_per_bk - num_grab_recv_warp;
	uint offload_recv_start_warp = 0;
	uint offload_recv_end_warp = num_offload_recv_warp;
	uint grab_recv_start_warp = num_offload_recv_warp;
	uint grab_recv_end_warp = num_offload_recv_warp + num_grab_recv_warp;

	//offload channel
	{
		//num receiver on the server side
		int exit_counter = client_block_size * total_client_bk;
		uint num_sender_per_bk = client_block_size / 32;
		int leader_warp_id = server_block_size / 32 - 1;

		init_gbc(&(gbc_pack.gbc[CHANNEL_OFFLOAD]), MSG_SIZE_OFFLOAD,
				exit_counter, offload_recv_start_warp * 32,
				offload_recv_end_warp * 32, num_sender_per_bk, leader_warp_id,
				-1, WORK_BUFF_SIZE_OFFLOAD);

	}

	//grab channel
	{
		//num receiver on the server side
#if SHARE_GRAB == 0
		int leader_warp_id = server_block_size / 32 - 2;
#else
		int leader_warp_id = server_block_size / 32 - 1;
#endif

		int exit_counter = (num_offload_recv_warp * 32) * NUM_RECEIVER;
		uint num_sender_per_bk = num_offload_recv_warp;

		init_gbc(&(gbc_pack.gbc[CHANNEL_GRAB]), MSG_SIZE_GRAB, exit_counter,
				grab_recv_start_warp * 32, grab_recv_end_warp * 32,
				num_sender_per_bk, leader_warp_id, -1, WORK_BUFF_SIZE_GRAB);

	}

	//reply channel
	{
		//num receiver on the server side
		int replyer_warp_id = server_block_size / 32 - NUM_LEADER + 1;
		int exit_counter = (num_grab_recv_warp * 32) * NUM_RECEIVER;
		uint num_sender_per_bk = num_grab_recv_warp;

		init_gbc(&(gbc_pack.gbc[CHANNEL_REPLY]), MSG_SIZE_REPLY, exit_counter,
				-1, -1, num_sender_per_bk, -1, replyer_warp_id,
				WORK_BUFF_SIZE_REPLY);

	}
	//unlock channel
	{
		//num receiver on the server side
		int unlocker_warp_id = server_block_size / 32 - NUM_LEADER;
		int exit_counter = (num_offload_recv_warp * 32) * NUM_RECEIVER;
		uint num_sender_per_bk = num_offload_recv_warp;

		init_gbc(&(gbc_pack.gbc[CHANNEL_UNLOCK]), MSG_SIZE_UNLOCK, exit_counter,
				-1, -1, num_sender_per_bk, -1, unlocker_warp_id,
				WORK_BUFF_SIZE_UNLOCK);

	}

	///////////////////////////////////////
	///////////////////////////////////////
}

__device__ void termination_process(gbc_pack_t gbc_pack) {

	if (threadIdx.x < gbc_pack.gbc[CHANNEL_OFFLOAD].receiver_end) {
#if GRAB_VERSION == BASE_SEND
		base_exit(gbc_pack.gbc[CHANNEL_GRAB]);
#elif GRAB_VERSION == ADV_SEND
		adv_exit(gbc_pack.gbc[CHANNEL_GRAB], CHANNEL_GRAB, _grab_coal_buf_data);
#endif

#if UNLOCK_VERSION == BASE_SEND
		base_exit(gbc_pack.gbc[CHANNEL_UNLOCK]);
#elif UNLOCK_VERSION == ADV_SEND
		adv_exit(gbc_pack.gbc[CHANNEL_UNLOCK], CHANNEL_UNLOCK, _unlock_coal_buf_data);
#endif
	} else if (threadIdx.x < gbc_pack.gbc[CHANNEL_GRAB].receiver_end) {

#if REPLY_VERSION == BASE_SEND
		base_exit(gbc_pack.gbc[CHANNEL_REPLY]);
#elif REPLY_VERSION == ADV_SEND
		adv_exit(gbc_pack.gbc[CHANNEL_REPLY], CHANNEL_REPLY, _reply_coal_buf_data);
#endif
	}

}

__device__ void main_worker(gbc_pack_t gbc_pack, ALG_ARG_DEF) {

//generic
	uint m_warp_id = thread_id_x() / 32;
	int receiver_id = blockIdx.x;
	do {
		SEND_VAR_DEF;
		uint pending_work = 0;
		uint stage = 0;
#if GRAB_VERSION == BASE_SEND
		uint saved_write_ptr = NOT_FOUND;
#endif
		//get work buff_status
		uint buff_status;
		buff_status = gc_get_work_buff(CHANNEL_OFFLOAD, m_warp_id, 0);
		if (buff_status != NOT_FOUND) {
			if (buff_status == GC_TERM) {
				termination_process(gbc_pack);
				asm volatile("exit;");
			} else {
				uint idx = extract_bits(buff_status, 0, AB_SIZE_START_BIT);
				uint size = extract_bits(buff_status, AB_SIZE_START_BIT,
						AB_SIZE_NUM_BIT);

				//setup pending work mask
				stage = (get_lane_id() < size) ? 1 : 0;
				pending_work = vote_ballot(stage);
				if (get_lane_id() < size) {
					//get msg data
					get_offload_data(gbc_pack, receiver_id, idx, SEND_ARG);
#if 0
					{
						uint account_id1 = val0;
						uint account_id2 = val1;
						uint dst2 = (account_id2 / DIV_FACTOR) % NUM_RECEIVER;
						uint lock_id1 = get_local_lock_id(account_id1);
						uint lock_id2 = get_local_lock_id(account_id2);
						atomicAdd(
								&(local_request[receiver_id * NUM_LOCK
										+ lock_id1]), 1);
						atomicAdd(&(remote_request[dst2 * NUM_LOCK + lock_id2]),
								1);
					}
#endif

				}
				//reset buff_status
				gc_receiver_reset_mp_status(gbc_pack, CHANNEL_OFFLOAD,
						m_warp_id, 0);

			}
		}
		uint account_id1 = val0;
		uint account_id2 = val1;
		uint dst2 = (account_id2 / DIV_FACTOR) % NUM_RECEIVER;
		uint lock_id1 = get_local_lock_id(account_id1);
		uint lock_id2 = get_local_lock_id(account_id2);
		bool one_lock = (receiver_id == dst2) && (lock_id1 == lock_id2);
		while (pending_work != 0) {
			//need to grab the lock from the next server
			//send grab msg
			uint valid_msg = 0; //send unlock to next phase
			uint grab_msg;
			if (stage == 1) {
				if (one_lock) {
					atomicAdd(&(lock_struct[lock_id1]), 1);
					stage = 3;
				} else {
					//lower 14, lock id
					grab_msg = lock_id2;
					//mid 10, tid
					grab_msg = set_bits(grab_msg, thread_id_x(), 14, 10);
					//upper 5, receiver id
					grab_msg = set_bits(grab_msg, receiver_id, 24, 5);
					valid_msg = 1;
				}
			}

#if GRAB_VERSION == BASE_SEND
			if (base_send(gbc_pack.gbc[CHANNEL_GRAB], dst2, valid_msg,
					saved_write_ptr, _grab_tail_ptr_cpy, grab_msg, 0, 0)) {
				//success
				if (valid_msg) {
					stage = 2;
				}
			}
#elif GRAB_VERSION == ADV_SEND

			if (adv_send_msg(gbc_pack.gbc[CHANNEL_GRAB],CHANNEL_GRAB,dst2, valid_msg,
							_grab_coal_buf_data, grab_msg, 0, 0)) {
				//success
				if (valid_msg) {
					stage = 2;
				}
			}
#endif
			if (stage == 2) {
				//see if lock2 is grabbed
				if ((p1_signal[m_warp_id] & (1 << get_lane_id())) != 0) {
					//reserve
					atomicAdd(&(lock_struct[lock_id1]), 1);
					stage = 3;
				}
			}

			valid_msg = 0; //send unlock to next phase
			if (stage == 3) {
				//grabbed lock 2
				//try get lock1
				bool acquired = ((atomicOr(&(lock_struct[lock_id1]), LOCK_MASK)
						& LOCK_MASK) == 0);

				if (acquired) {
					//un-reserve
					atomicSub(&(lock_struct[lock_id1]), 1);
					//do critical section
					critcal_section(SEND_ARG, ALG_ARG);
					//unlock
					//unlock lock1
					atomicAnd(&(lock_struct[lock_id1]), ~LOCK_MASK);
					if (!one_lock) {
						valid_msg = 1;
					}
					//no work anymore
					stage = 0;
					//unset signal
					atomicAnd(&(p1_signal[m_warp_id]), ~(1 << get_lane_id()));
				}
			}

			//send unlock
#if UNLOCK_VERSION == BASE_SEND
			uint _unlock_saved_write_ptr = NOT_FOUND;
			do {
				if (base_send(gbc_pack.gbc[CHANNEL_UNLOCK], dst2, valid_msg,
						_unlock_saved_write_ptr, _unlock_tail_ptr_cpy, lock_id2,
						0, 0)) {
					valid_msg = 0;
				}
			} while (vote_ballot(valid_msg) != 0);
#elif UNLOCK_VERSION == ADV_SEND

			do {
				if (adv_send_msg(gbc_pack.gbc[CHANNEL_UNLOCK],CHANNEL_UNLOCK, dst2, valid_msg,
								_unlock_coal_buf_data, lock_id2,
								0, 0)) {
					valid_msg = 0;
				}
			}while (vote_ballot(valid_msg) != 0);

#endif

			//vote again
			pending_work = vote_ballot((stage != 0) ? 1 : 0);
			//if not work at all, reset local buffer
			if (pending_work == 0) {
				//reset work buff
				gc_reset_work_buff(CHANNEL_OFFLOAD, m_warp_id, 0);
			}

		}
	} while (1);
	asm volatile("exit;");
}

__device__ void process_buffer_grab_worker(gbc_pack_t gbc_pack, uint m_warp_id,
		uint buf_id, uint& pending_work, uint& data) {
	int receiver_id = blockIdx.x;
	if (pending_work == 0) {
		uint status;
		status = gc_get_work_buff(CHANNEL_GRAB, m_warp_id, buf_id);
		if (status != NOT_FOUND) {
			if (status == GC_TERM) {
				termination_process(gbc_pack);
				asm volatile("exit;");
			} else {
				uint idx = extract_bits(status, 0, AB_SIZE_START_BIT);
				uint size = extract_bits(status, AB_SIZE_START_BIT,
						AB_SIZE_NUM_BIT);

				uint has_work = 0;
				//vertical version
				if (get_lane_id() < size) {
					data =
							gbc_pack.gbc[CHANNEL_GRAB].queue_data[0][receiver_id][idx
									+ get_lane_id()];
					has_work = 1;
				}
				//reset buff_status
				gc_receiver_reset_mp_status(gbc_pack, CHANNEL_GRAB, m_warp_id,
						buf_id);
				pending_work = vote_ballot(has_work);
			}
		}
	}

	if (pending_work != 0) {
		uint has_work = extract_bits(pending_work, get_lane_id(), 1);
		uint valid_msg = 0; //send to next phase
		uint lock_id = extract_bits(data, 0, 14);

		if (has_work) {
			//see if the lock is unreserved
			if (extract_bits(lock_struct[lock_id], 0, 16) == 0) {
				if ((atomicOr(&(lock_struct[lock_id]), LOCK_MASK) & LOCK_MASK)
						== 0) {
					//grabbed the lock
					valid_msg = 1;
					has_work = 0;
				}
			}
		}

		uint dst = extract_bits(data, 24, 5);

#if REPLY_VERSION == BASE_SEND
		uint _reply_saved_write_ptr = NOT_FOUND;
		do {
			if (base_send(gbc_pack.gbc[CHANNEL_REPLY], dst, valid_msg,
					_reply_saved_write_ptr, _reply_tail_ptr_cpy, data, 0, 0)) {
				valid_msg = 0;
			}
		} while (vote_ballot(valid_msg) != 0);
#elif REPLY_VERSION == ADV_SEND

		do {
			if (adv_send_msg(gbc_pack.gbc[CHANNEL_REPLY],CHANNEL_REPLY, dst, valid_msg,
							_reply_coal_buf_data, data, 0, 0)) {
				valid_msg = 0;
			}
		}while (vote_ballot(valid_msg) != 0);

#endif

		//vote again
		pending_work = vote_ballot(has_work);
		//if not work at all, reset local buffer

		if (pending_work == 0) {
			//reset work buff
			gc_reset_work_buff(CHANNEL_GRAB, m_warp_id, buf_id);
			return;
		}

	}
}

__device__ void grab_worker(gbc_pack_t gbc_pack) {

//generic
	uint m_warp_id = thread_id_x() / 32;
#if WORK_BUFF_SIZE_GRAB > 0
	uint pending_work0 = 0;
	uint data_buf0;
#endif

#if WORK_BUFF_SIZE_GRAB > 1
	uint pending_work1 = 0;
	uint data_buf1;
#endif

#if WORK_BUFF_SIZE_GRAB > 2
	uint pending_work2 = 0;
	uint data_buf2;
#endif

#if WORK_BUFF_SIZE_GRAB > 3
	uint pending_work3 = 0;
	uint data_buf3;
#endif

#if WORK_BUFF_SIZE_GRAB > 4
	uint pending_work4 = 0;
	uint data_buf4;
#endif

#if WORK_BUFF_SIZE_GRAB > 5
	uint pending_work5 = 0;
	uint data_buf5;
#endif

#if WORK_BUFF_SIZE_GRAB > 6
	uint pending_work6 = 0;
	uint data_buf6;
#endif

#if WORK_BUFF_SIZE_GRAB > 7
	uint pending_work7 = 0;
	uint data_buf7;
#endif

#if WORK_BUFF_SIZE_GRAB > 8
	assert(false);
#endif

	while (1) {
#if WORK_BUFF_SIZE_GRAB > 0
		process_buffer_grab_worker(gbc_pack, m_warp_id, 0, pending_work0,
				data_buf0);
#endif

#if WORK_BUFF_SIZE_GRAB > 1
		process_buffer_grab_worker(gbc_pack, m_warp_id, 1, pending_work1,
				data_buf1);
#endif

#if WORK_BUFF_SIZE_GRAB > 2
		process_buffer_grab_worker(gbc_pack, m_warp_id, 2, pending_work2,
				data_buf2);
#endif

#if WORK_BUFF_SIZE_GRAB > 3
		process_buffer_grab_worker(gbc_pack, m_warp_id, 3, pending_work3,
				data_buf3);
#endif

#if WORK_BUFF_SIZE_GRAB > 4
		process_buffer_grab_worker(gbc_pack, m_warp_id, 4, pending_work4, data_buf4);
#endif

#if WORK_BUFF_SIZE_GRAB > 5
		process_buffer_grab_worker(gbc_pack, m_warp_id, 5, pending_work5, data_buf5);
#endif

#if WORK_BUFF_SIZE_GRAB > 6
		process_buffer_grab_worker(gbc_pack, m_warp_id, 6, pending_work6, data_buf6);
#endif

#if WORK_BUFF_SIZE_GRAB > 7
		process_buffer_grab_worker(gbc_pack, m_warp_id, 7, pending_work7, data_buf7);
#endif
	}

	asm volatile("exit;");
}

__global__ void server_two_phase(gbc_pack_t gbc_pack, ALG_ARG_DEF) {
	if (thread_id_x() == 0) {
		for (int i = 0; i < NUM_CHANNEL; i++) {
			total_iter[i] = 0;
			total_req[i] = 0;
		}
	}

	init_server();
	init_recv(gbc_pack);

#if  GRAB_VERSION == BASE_SEND
	init_base(_grab_tail_ptr_cpy);
#elif GRAB_VERSION == ADV_SEND
	init_adv(gbc_pack.gbc[CHANNEL_GRAB],CHANNEL_GRAB);
#endif
#if  REPLY_VERSION == BASE_SEND
	init_base(_reply_tail_ptr_cpy);
#elif REPLY_VERSION == ADV_SEND
	init_adv(gbc_pack.gbc[CHANNEL_REPLY],CHANNEL_REPLY);
#endif

#if  UNLOCK_VERSION == BASE_SEND
	init_base(_unlock_tail_ptr_cpy);
#elif UNLOCK_VERSION == ADV_SEND
	init_adv(gbc_pack.gbc[CHANNEL_UNLOCK],CHANNEL_UNLOCK);
#endif

	/////////////////////////////
	/*
	 * leader
	 */

	gc_receiver_leader(gbc_pack);
	//unlocker
	gc_receiver_leader_worker(gbc_pack.gbc[CHANNEL_UNLOCK], gc_unlock_func);
	//replyer
	gc_receiver_leader_worker(gbc_pack.gbc[CHANNEL_REPLY], gc_reply_func);

	//////////////////////////////////

	///////////////////////////////
	/*
	 * regular
	 */
	if (threadIdx.x < gbc_pack.gbc[CHANNEL_OFFLOAD].receiver_end) {
		main_worker(gbc_pack, ALG_ARG);
	} else {
		grab_worker(gbc_pack);
	}

}

#endif /* TWO_PHASE_SERVER_H_ */
