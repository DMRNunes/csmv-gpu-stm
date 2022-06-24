/*
 * server_comm.h
 *
 *  Created on: May 29, 2018
 *      Author: redudie
 */

#ifndef SERVER_COMM_H_
#define SERVER_COMM_H_
#define NOT_FOUND 0xffffffff
//logic instructions
__device__ __forceinline__ unsigned vote_ballot(uint ballot) {
	return __ballot_sync(__activemask(),ballot);
}

__device__ __forceinline__ void break_pt() {
	asm volatile (
			"brkpt;"
	);
}

__device__ __forceinline__ void barrier(uint name, uint thread_count) {
	asm volatile (
			"bar.sync %0,%1;"
			::"r" (name), "r"(thread_count)
	);
}

__device__ __forceinline__ unsigned load_cg(uint* addr) {
	uint ret_val;
	asm volatile (
			"ld.global.cg.u32 %0, [%1];"
			: "=r" (ret_val) : "l"(addr)
	);
	return ret_val;
}

__device__ __forceinline__ void store_cg(uint* addr, uint val) {
	asm volatile (
			"st.global.cg.f32 [%0],%1;"
			::"l" (addr), "r"(val)
	);
}

__device__ __forceinline__ unsigned find_ms_bit(uint bit_mask) {
	uint ret_val;
	asm volatile (
			"bfind.u32 %0, %1;"
			: "=r" (ret_val) : "r"(bit_mask)
	);
	return ret_val;
}

__device__ __forceinline__ unsigned count_bit(uint bit_mask) {
	uint ret_val;
	asm volatile (
			"popc.b32 %0, %1;"
			: "=r" (ret_val) : "r"(bit_mask)
	);
	return ret_val;
}

__device__ __forceinline__ unsigned extract_bits(uint bit_mask, uint start_pos,
		uint len) {
	uint ret_val;
	asm volatile (
			"bfe.u32 %0, %1, %2, %3;"
			: "=r" (ret_val) : "r"(bit_mask), "r" (start_pos), "r"(len)
	);
	return ret_val;
}

__device__ __forceinline__ unsigned set_bits(uint bit_mask, uint val,
		uint start_pos, uint len) {
	uint ret_val;
	asm volatile (
			"bfi.b32 %0, %1, %2, %3, %4;"
			: "=r" (ret_val) :"r" (val), "r"(bit_mask), "r" (start_pos), "r"(len)
	);
	return ret_val;
}

__device__ __forceinline__ unsigned find_nth_bit(uint bit_mask, uint base,
		uint offset) {
	uint ret_val;
	asm volatile (
			"fns.b32 %0, %1, %2, %3;"
			: "=r" (ret_val) :"r" (bit_mask), "r"(base), "r" (offset)
	);
	return ret_val;
}

__forceinline__ __device__ unsigned get_lane_id() {
	unsigned ret;
	asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
	return ret;
}



__forceinline__ __device__ unsigned get_clock32() {
	unsigned ret;
	asm volatile ("mov.u32 %0, %clock;" : "=r"(ret));
	return ret;
}

__forceinline__ __device__ unsigned shuffle_idx(uint value, uint src_lane) {
	uint ret_val;
	asm (
			"shfl.sync.idx.b32 %0, %1, %2, %3, %4;\n\t"
			: "=r" (ret_val) : "r"(value), "r"(src_lane), "r"(0x1F), "r"(0xffffffff)
	);
	return ret_val;
}

__forceinline__ __device__ unsigned shuffle_up(uint value, uint src_lane) {
	uint ret_val;
	asm (
			"shfl.sync.up.b32 %0, %1, %2, %3;\n\t"
			: "=r" (ret_val) : "r"(value), "r"(src_lane), "r"(0x1F)
	);
	return ret_val;

}

__forceinline__ __device__ unsigned shuffle_xor(uint value, uint src_lane) {
	uint ret_val;
	asm (
			"shfl.sync.bfly.b32 %0, %1, %2, %3;\n\t"
			: "=r" (ret_val) : "r"(value), "r"(src_lane), "r"(0x1F)
	);
	return ret_val;
}

__forceinline__ __device__ unsigned thread_id_x() {
	return threadIdx.x;
}

__forceinline__ __device__ unsigned block_id_x() {
	return blockIdx.x;
}

__forceinline__ __device__ unsigned block_dim_x() {
	return blockDim.x;
}

__forceinline__ __device__ void sync_threads() {
	__syncthreads();
}

__forceinline__ __device__ void mem_fence_block() {
	__threadfence_block();
}

__forceinline__ __device__ uint __atomic_or(uint* addr, uint value) {
	return atomicOr(addr, value);
}

__forceinline__ __device__ uint __atomic_and(uint* addr, uint value) {
	return atomicAnd(addr, value);
}

__forceinline__ __device__ uint __atomic_add(uint* addr, uint value) {
	return atomicAdd(addr, value);
}

__forceinline__ __device__ uint __atomic_sub(uint* addr, uint value) {
	return atomicSub(addr, value);
}

__forceinline__ __device__ uint __atomic_exch(uint* addr, uint value) {
	return atomicExch(addr, value);
}

#define PTR_BITS 12
#define STATUS_BITS (32-PTR_BITS-PTR_BITS-2)
#define IN_WORK_MASK (1 << (STATUS_BITS-1))
#define SIZE_BITS (STATUS_BITS -2)
#define STATUS_SIZE_MAX ((1<<SIZE_BITS)-1)
#define SIZE_MASK ((1 << (SIZE_BITS))-1)
#define LOCKED_BIT (1<<(STATUS_BITS-2))
#define NULL_PTR ((1<<(PTR_BITS))-1)

//header struct writer
__device__ __forceinline__ unsigned header_set_status(uint header, uint val) {
	header = set_bits(header, val, 0, STATUS_BITS);
	return header;
}

__device__ __forceinline__ unsigned header_set_size(uint header, uint val) {
	header = set_bits(header, val, 0, SIZE_BITS);
	return header;
}

__device__ __forceinline__ unsigned header_set_tail_ptr(uint header, uint val) {
	header = set_bits(header, val, STATUS_BITS, PTR_BITS);
	return header;
}

__device__ __forceinline__ unsigned header_set_tail_type(uint header,
		uint val) {
	header = set_bits(header, val, (STATUS_BITS + PTR_BITS), 1);
	return header;
}

__device__ __forceinline__ unsigned header_set_head_ptr(uint header, uint val) {
	header = set_bits(header, val, (STATUS_BITS + PTR_BITS + 1), PTR_BITS);
	return header;
}

__device__ __forceinline__ unsigned header_set_head_type(uint header,
		uint val) {
	header = set_bits(header, val, (STATUS_BITS + PTR_BITS + PTR_BITS + 1), 1);
	return header;
}

//header struct reader
__device__ __forceinline__ unsigned header_get_status(uint header) {
	uint val = extract_bits(header, 0, STATUS_BITS);
	return val;
}

__device__ __forceinline__ unsigned header_get_size(uint header) {
	uint val = extract_bits(header, 0, SIZE_BITS);
	return val;
}

__device__ __forceinline__ unsigned header_get_tail_ptr(uint header) {
	uint val = extract_bits(header, STATUS_BITS, PTR_BITS);
	return val;
}

__device__ __forceinline__ unsigned header_get_tail_type(uint header) {
	uint val = extract_bits(header, (STATUS_BITS + PTR_BITS), 1);
	return val;
}

__device__ __forceinline__ unsigned header_get_head_ptr(uint header) {
	uint val = extract_bits(header, (STATUS_BITS + PTR_BITS + 1), PTR_BITS);
	return val;
}

__device__ __forceinline__ unsigned header_get_head_type(uint header) {
	uint val = extract_bits(header, (STATUS_BITS + PTR_BITS + PTR_BITS + 1), 1);
	return val;
}

__device__ __forceinline__ unsigned slot_meta_set_next_ptr(uint meta_data,
		uint val) {
	meta_data = set_bits(meta_data, val, 0, PTR_BITS);
	return meta_data;
}

__device__ __forceinline__ unsigned slot_meta_set_next_type(uint meta_data,
		uint val) {
	meta_data = set_bits(meta_data, val, PTR_BITS, 1);
	return meta_data;
}

__device__ __forceinline__ unsigned slot_meta_get_next_ptr(uint meta_data) {
	uint val = extract_bits(meta_data, 0, PTR_BITS);
	return val;
}

__device__ __forceinline__ unsigned slot_meta_get_next_type(uint meta_data) {
	uint val = extract_bits(meta_data, PTR_BITS, 1);
	return val;
}






#endif /* SERVER_COMM_H_ */
