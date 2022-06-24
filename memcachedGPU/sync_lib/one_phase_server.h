 #ifndef ONE_PHASE_SERVER_H_
#define ONE_PHASE_SERVER_H_

__shared__ uint lock[NUM_LOCK];

__device__ void init_server() {
	for (int i = threadIdx.x; i < NUM_LOCK; i += blockDim.x) {
		lock[i] = 0;
	}
	__syncthreads();
}
//one phase server
void create_gbc(gbc_pack_t & gbc_pack, int total_client_bk,
		int client_block_size, int server_block_size) {

	///////////////////////////////////////////////
	//gbc init
#define NUM_LEADER 1

	int num_receiver_per_bk = server_block_size / 32;
	num_receiver_per_bk -= NUM_LEADER;

	uint num_offload_recv_warp = num_receiver_per_bk;
	uint offload_recv_start_warp = 0;
	uint offload_recv_end_warp = num_offload_recv_warp;

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

	///////////////////////////////////////
}

__device__ void process_buffer_main_worker(gbc_pack_t gbc_pack,
		uint m_warp_id, uint buf_id, SEND_ARG_DEF_REF, uint& stage,
		uint& lock_id, SERV_ARG_DEF) {
	int receiver_id = blockIdx.x;
//printf("S%d: stage %d\n", thread_id_x(), stage);
	uint pending_work = vote_ballot((stage != 0) ? 1 : 0);
	if (pending_work == 0) {
		//get work buff_status
		uint buff_status;
		buff_status = gc_get_work_buff(CHANNEL_OFFLOAD, m_warp_id, buf_id);
		if (buff_status != NOT_FOUND) {
			if (buff_status == GC_TERM) {
				asm volatile("exit;");
			} else {
				uint idx = extract_bits(buff_status, 0, AB_SIZE_START_BIT);
				uint size = extract_bits(buff_status, AB_SIZE_START_BIT,
						AB_SIZE_NUM_BIT);
#if PRINT_DEBUG == 0
				if(get_lane_id()==0) printf("\t\tS%d: %d %x %d %d\n", m_warp_id, buf_id, buff_status, idx, size);
#endif
//if(thread_id_x()%32==0) printf("\t\tS%d: idx %d size %d laneid %d\n", thread_id_x()/32, idx, size, get_lane_id());
				//setup pending work mask
				stage = (get_lane_id() < size) ? 1 : 0;
				pending_work = vote_ballot(stage);
				if (get_lane_id() < size) {
					//get msg data
					get_offload_data(gbc_pack, receiver_id, idx, SEND_ARG);
					lock_id = get_local_lock_id(val0);
				}
				//reset buff_status
				gc_receiver_reset_mp_status(gbc_pack, CHANNEL_OFFLOAD, m_warp_id,
						buf_id);

			}
		}
		//if(thread_id_x()%32==0) printf("\t\tS%d: not found buffer\n", thread_id_x()/32);				

	}

	if (pending_work != 0) {

		if (stage == 1) {
			//uint account_id = val0;
			//bool acquired = (atomicCAS(&(lock[lock_id]), 0, 1) == 0);
			//if (acquired) {
				//do critical section
				critcal_section(SERV_ARG, SEND_ARG); //, ALG_ARG);
				//unlock
				mem_fence_block();
				//atomicExch(&(lock[lock_id]), 0);
				//no work anymore
				stage = 0;
			//}

		}
		//vote again
		pending_work = vote_ballot((stage != 0) ? 1 : 0);
		//if not work at all, reset local buffer
		if (pending_work == 0) {
			//reset work buff
			gc_reset_work_buff(CHANNEL_OFFLOAD, m_warp_id, buf_id);
		}

	}
}
/*
__device__ void main_worker(gbc_pack_t gbc_pack, ALG_ARG_DEF) {

//generic
	uint m_warp_id = thread_id_x() / 32;

#if WORK_BUFF_SIZE_OFFLOAD > 0
	VAR_BUF_DEF0
	uint stage_buf0 = 0;
	uint lock_id0;
#endif

#if WORK_BUFF_SIZE_OFFLOAD > 1
	VAR_BUF_DEF1
	uint stage_buf1 = 0;
	uint lock_id1;
#endif

#if WORK_BUFF_SIZE_OFFLOAD > 2
	VAR_BUF_DEF2
	uint stage_buf2 = 0;
	uint lock_id2;
#endif
#if WORK_BUFF_SIZE_OFFLOAD > 3
	assert(false);
#endif

	while (1) {
#if WORK_BUFF_SIZE_OFFLOAD > 0
		process_buffer_main_worker(gbc_pack, m_warp_id, 0, VAR_BUF0,
				stage_buf0,lock_id0);
#endif

#if WORK_BUFF_SIZE_OFFLOAD > 1
		process_buffer_main_worker(gbc_pack, m_warp_id, 1, VAR_BUF1,
				stage_buf1,lock_id1);
#endif

#if WORK_BUFF_SIZE_OFFLOAD > 2
		process_buffer_main_worker(gbc_pack, m_warp_id, 2, VAR_BUF2,
				stage_buf2,lock_id2);
#endif

	}
}
*/

/*__global__ void server_one_phase(gbc_pack_t gbc_pack, ALG_ARG_DEF) {

	init_server();
	init_recv(gbc_pack);
	/////////////////////////////
	/*
	 * leader
	 /

	gc_receiver_leader(gbc_pack);
	//////////////////////////////////

	///////////////////////////////
	/*
	 * regular
	 /

	main_worker(gbc_pack, ALG_ARG);
	///////////////////////////////
}
*/
#endif /* ONE_PHASE_SERVER_H_ */