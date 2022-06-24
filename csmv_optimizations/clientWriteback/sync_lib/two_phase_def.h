/*
 * server.h
 *
 *  Created on: Oct 13, 2018
 *      Author: redudie
 */

#ifndef SERVER_H_
#define SERVER_H_



#define NUM_CHANNEL 4
//work buffer set up
#define WORK_BUFF_SIZE_OFFLOAD 1
#define WORK_BUFF_SIZE_GRAB 4
#define WORK_BUFF_SIZE_REPLY 1
#define WORK_BUFF_SIZE_UNLOCK 1

#define MSG_SIZE_GRAB 1
#define MSG_SIZE_REPLY 1
#define MSG_SIZE_UNLOCK 1

#if WORK_BUFF_SIZE_OFFLOAD > WORK_BUFF_SIZE_GRAB
#define WORK_BUFF_SIZE_MAX WORK_BUFF_SIZE_OFFLOAD
#else
#define WORK_BUFF_SIZE_MAX WORK_BUFF_SIZE_GRAB
#endif

#define NUM_GRAB_WARP 4

#define CHANNEL_OFFLOAD 0
#define CHANNEL_GRAB 1
#define CHANNEL_REPLY 2
#define CHANNEL_UNLOCK 3

#define NUM_LOCK (8192)

#endif /* SERVER_H_ */
