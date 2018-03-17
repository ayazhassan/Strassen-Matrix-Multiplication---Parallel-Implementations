
#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_
	
// Thread block size
#define BLOCK_SIZE 16
#define TPB 16
#define TG 2
#define xBLOCKS 512
#define yBLOCKS 512
#define TILE_X 32
#define TILE_Y 16
#define BLOCK_X TILE_X
#define BLOCK_Y 16


#define Asize 2048
#define ASIZE Asize
#define N ASIZE
#define MULTILEVEL
#define BLOCK_THRESHOLD 2048 //BLOCK_X * 2
#define TAU1 1024
#define TAU2 2048

#define WA Asize // Matrix A width
#define HA Asize // Matrix A height
#define WB Asize // Matrix B width
#define HB Asize  // Matrix B height
#define WC Asize  // Matrix C width 
#define HC Asize  // Matrix C height

//#define STRASSEN_WITH_GPGPUCOMPILER_KERNELS

#endif // _MATRIXMUL_H_


