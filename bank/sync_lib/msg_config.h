
#ifndef GB_CONFIG_H_
#define GB_CONFIG_H_

//let reply and unlock share the same channel
#define SHARE_GRAB 1


//gb config
#define GC_BUFF_SIZE (4096)
#define DIV_FACTOR 32
#define DIV_SHIFT 5
#define RESET_THD 992
#define NUM_RESET_BUF 8




//server config
#define NUM_LOCK_GLOBAL (NUM_RECEIVER * NUM_LOCK)




//no need to change
#define GC_TERM 0xefffffff
#define GC_NUM_MASK (GC_BUFF_SIZE/32)
#define NOT_FOUND 0xffffffff
#define PRINT_DEBUG 1
#define EXIT_REQUEST 0xfffffffc
#define AB_RESET_NUM_BIT (3)
#define AB_SIZE_NUM_BIT 6
#define AB_RESET_START_BIT (32-AB_RESET_NUM_BIT)
#define AB_SIZE_START_BIT (AB_RESET_START_BIT-6)

#define TAIL_UPDATE_THD 500

#endif /* GB_CONFIG_H_ */
