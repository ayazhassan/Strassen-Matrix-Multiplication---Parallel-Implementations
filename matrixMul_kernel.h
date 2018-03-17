
#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixMul.h"

#define COALESCED_NUM 16
#define blockDimX 256
#define blockDimY 1
#define gridDimX (gridDim.x)
#define gridDimY (gridDim.y)
#define idx (blockIdx.x*blockDimX+threadIdx.x)
#define idy (blockIdx.y*blockDimY+threadIdx.y)
#define bidy (blockIdx.y)
#define bidx (blockIdx.x)
#define tidx (threadIdx.x)
#define tidy (threadIdx.y)
#define merger_y 16
#define coalesced_idy (bidy/(COALESCED_NUM/(merger_y*blockDimY))*COALESCED_NUM)
#define B(y,x) B[(y)*ASIZE+(x)]
#define C(y,x) C[(y)*ASIZE+(x)]
#define A(y,x) A[(y)*ASIZE+(x)]

__global__ void matmul_opt_restructured(float * A, float * B, float * C, int width, int widthA, int widthB, int widthC)
{
        __shared__ float shared_0[16][65];
        int i;
        float sum_0;
        float sum_1;
        float sum_2;
        float sum_3;
        float sum_4;
        float sum_5;
        float sum_6;
        float sum_7;
        float sum_8;
        float sum_9;
        float sum_10;
        float sum_11;
        float sum_12;
        float sum_13;
        float sum_14;
        float sum_15;
        sum_0=0;
        sum_1=0;
        sum_2=0;
        sum_3=0;
        sum_4=0;
        sum_5=0;
        sum_6=0;
        sum_7=0;
        sum_8=0;
        sum_9=0;
        sum_10=0;
        sum_11=0;
        sum_12=0;
        sum_13=0;
        sum_14=0;
        sum_15=0;
        for (i=0; i<width; i=(i+64))
        {
                int it_1;

        shared_0[(tidx/64)+0][(tidx%64)]=A[(((bidy*16)+tidy)+(tidx/64)+0) * widthA + (i+(tidx%64))];
        shared_0[(tidx/64)+4][(tidx%64)]=A[(((bidy*16)+tidy)+(tidx/64)+4) * widthA + (i+(tidx%64))];
        shared_0[(tidx/64)+8][(tidx%64)]=A[(((bidy*16)+tidy)+(tidx/64)+8) * widthA + (i+(tidx%64))];
        shared_0[(tidx/64)+12][(tidx%64)]=A[(((bidy*16)+tidy)+(tidx/64)+12) * widthA + (i+(tidx%64))];
                __syncthreads();
                #pragma unroll
                for (it_1=0; it_1<64; it_1=(it_1+1))
                {
                        float a_0;
                        float a_1;
                        float a_2;
                        float a_3;
                        float a_4;
                        float a_5;
                        float a_6;
                        float a_7;
                        float a_8;
                        float a_9;
                        float a_10;
                        float a_11;
                        float a_12;
                        float a_13;
                        float a_14;
                        float a_15;
                        float b;
                        a_0=shared_0[0][it_1];
                        a_1=shared_0[1][it_1];
                        a_2=shared_0[2][it_1];
                        a_3=shared_0[3][it_1];
                        a_4=shared_0[4][it_1];
                        a_5=shared_0[5][it_1];
                        a_6=shared_0[6][it_1];
                        a_7=shared_0[7][it_1];
                        a_8=shared_0[8][it_1];
                        a_9=shared_0[9][it_1];
                        a_10=shared_0[10][it_1];
                        a_11=shared_0[11][it_1];
                        a_12=shared_0[12][it_1];
                        a_13=shared_0[13][it_1];
                        a_14=shared_0[14][it_1];
                        a_15=shared_0[15][it_1];
                        b=B[(it_1+i)*widthB + idx]; //B((it_1+i), idx);
                        sum_0+=(a_0*b);
                        sum_1+=(a_1*b);
                        sum_2+=(a_2*b);
                        sum_3+=(a_3*b);
                        sum_4+=(a_4*b);
                        sum_5+=(a_5*b);
                        sum_6+=(a_6*b);
                        sum_7+=(a_7*b);
                        sum_8+=(a_8*b);
                        sum_9+=(a_9*b);
                        sum_10+=(a_10*b);
                        sum_11+=(a_11*b);
                        sum_12+=(a_12*b);
                        sum_13+=(a_13*b);
                        sum_14+=(a_14*b);
                        sum_15+=(a_15*b);
                }
                __syncthreads();
        }
                C[(((bidy*16)+tidy)+0)*widthC + idx] = sum_0;
                C[(((bidy*16)+tidy)+1)*widthC + idx] = sum_1;
                C[(((bidy*16)+tidy)+2)*widthC + idx] = sum_2;
                C[(((bidy*16)+tidy)+3)*widthC + idx] = sum_3;
                C[(((bidy*16)+tidy)+4)*widthC + idx] = sum_4;
                C[(((bidy*16)+tidy)+5)*widthC + idx] = sum_5;
                C[(((bidy*16)+tidy)+6)*widthC + idx] = sum_6;
                C[(((bidy*16)+tidy)+7)*widthC + idx] = sum_7;
                C[(((bidy*16)+tidy)+8)*widthC + idx] = sum_8;
                C[(((bidy*16)+tidy)+9)*widthC + idx] = sum_9;
                C[(((bidy*16)+tidy)+10)*widthC + idx] = sum_10;
                C[(((bidy*16)+tidy)+11)*widthC + idx] = sum_11;
                C[(((bidy*16)+tidy)+12)*widthC + idx] = sum_12;
                C[(((bidy*16)+tidy)+13)*widthC + idx] = sum_13;
                C[(((bidy*16)+tidy)+14)*widthC + idx] = sum_14;
                C[(((bidy*16)+tidy)+15)*widthC + idx] = sum_15;
}


__global__ void matmul_opt(float * A, float * B, float * C, int width, int widthA, int widthB, int widthC)
{
	__shared__ float shared_0[16][17];
	int i;
	float sum_0;
	float sum_1;
	float sum_2;
	float sum_3;
	float sum_4;
	float sum_5;
	float sum_6;
	float sum_7;
	float sum_8;
	float sum_9;
	float sum_10;
	float sum_11;
	float sum_12;
	float sum_13;
	float sum_14;
	float sum_15;
	sum_0=0;
	sum_1=0;
	sum_2=0;
	sum_3=0;
	sum_4=0;
	sum_5=0;
	sum_6=0;
	sum_7=0;
	sum_8=0;
	sum_9=0;
	sum_10=0;
	sum_11=0;
	sum_12=0;
	sum_13=0;
	sum_14=0;
	sum_15=0;
	for (i=0; i<width; i=(i+16))
	{
		int it_1;
		
	shared_0[((tidx%16)+0)][(tidx/16)]=A[(((bidy*16)+tidy)+(tidx/16)) * widthA + (i+(tidx%16))]; //A((((bidy*16)+tidy)+(tidx/16)), (i+(tidx%16)));
		__syncthreads();
		#pragma unroll
		for (it_1=0; it_1<16; it_1=(it_1+1))
		{
			float a_0;
			float a_1;
			float a_2;
			float a_3;
			float a_4;
			float a_5;
			float a_6;
			float a_7;
			float a_8;
			float a_9;
			float a_10;
			float a_11;
			float a_12;
			float a_13;
			float a_14;
			float a_15;
			float b;
			a_0=shared_0[it_1][0];
			a_1=shared_0[it_1][1];
			a_2=shared_0[it_1][2];
			a_3=shared_0[it_1][3];
			a_4=shared_0[it_1][4];
			a_5=shared_0[it_1][5];
			a_6=shared_0[it_1][6];
			a_7=shared_0[it_1][7];
			a_8=shared_0[it_1][8];
			a_9=shared_0[it_1][9];
			a_10=shared_0[it_1][10];
			a_11=shared_0[it_1][11];
			a_12=shared_0[it_1][12];
			a_13=shared_0[it_1][13];
			a_14=shared_0[it_1][14];
			a_15=shared_0[it_1][15];
			b=B[(it_1+i)*widthB + idx]; //B((it_1+i), idx);
			sum_0+=(a_0*b);
			sum_1+=(a_1*b);
			sum_2+=(a_2*b);
			sum_3+=(a_3*b);
			sum_4+=(a_4*b);
			sum_5+=(a_5*b);
			sum_6+=(a_6*b);
			sum_7+=(a_7*b);
			sum_8+=(a_8*b);
			sum_9+=(a_9*b);
			sum_10+=(a_10*b);
			sum_11+=(a_11*b);
			sum_12+=(a_12*b);
			sum_13+=(a_13*b);
			sum_14+=(a_14*b);
			sum_15+=(a_15*b);
		}
		__syncthreads();
	}
	{
		C[(((bidy*16)+tidy)+0)*widthC + idx] = sum_0;
//		C((((bidy*16)+tidy)+0), idx)=sum_0;
	}
	{
		C[(((bidy*16)+tidy)+1)*widthC + idx] = sum_1;
//		C((((bidy*16)+tidy)+1), idx)=sum_1;
	}
	{
		C[(((bidy*16)+tidy)+2)*widthC + idx] = sum_2;
//		C((((bidy*16)+tidy)+2), idx)=sum_2;
	}
	{
		C[(((bidy*16)+tidy)+3)*widthC + idx] = sum_3;
//		C((((bidy*16)+tidy)+3), idx)=sum_3;
	}
	{
		C[(((bidy*16)+tidy)+4)*widthC + idx] = sum_4;
//		C((((bidy*16)+tidy)+4), idx)=sum_4;
	}
	{
		C[(((bidy*16)+tidy)+5)*widthC + idx] = sum_5;
//		C((((bidy*16)+tidy)+5), idx)=sum_5;
	}
	{
		C[(((bidy*16)+tidy)+6)*widthC + idx] = sum_6;
//		C((((bidy*16)+tidy)+6), idx)=sum_6;
	}
	{
		C[(((bidy*16)+tidy)+7)*widthC + idx] = sum_7;
//		C((((bidy*16)+tidy)+7), idx)=sum_7;
	}
	{
		C[(((bidy*16)+tidy)+8)*widthC + idx] = sum_8;
//		C((((bidy*16)+tidy)+8), idx)=sum_8;
	}
	{
		C[(((bidy*16)+tidy)+9)*widthC + idx] = sum_9;
//		C((((bidy*16)+tidy)+9), idx)=sum_9;
	}
	{
		C[(((bidy*16)+tidy)+10)*widthC + idx] = sum_10;
//		C((((bidy*16)+tidy)+10), idx)=sum_10;
	}
	{
		C[(((bidy*16)+tidy)+11)*widthC + idx] = sum_11;
//		C((((bidy*16)+tidy)+11), idx)=sum_11;
	}
	{
		C[(((bidy*16)+tidy)+12)*widthC + idx] = sum_12;
//		C((((bidy*16)+tidy)+12), idx)=sum_12;
	}
	{
		C[(((bidy*16)+tidy)+13)*widthC + idx] = sum_13;
//		C((((bidy*16)+tidy)+13), idx)=sum_13;
	}
	{
		C[(((bidy*16)+tidy)+14)*widthC + idx] = sum_14;
//		C((((bidy*16)+tidy)+14), idx)=sum_14;
	}
	{
		C[(((bidy*16)+tidy)+15)*widthC + idx] = sum_15;
//		C((((bidy*16)+tidy)+15), idx)=sum_15;
	}
}

#define CHECK_BANK_CONFLICTS 0
#if CHECK_BANK_CONFLICTS
#define AS(i, j) cutilBankChecker(((float*)&As[0][0]), (BLOCK_SIZE * i + 
j))
#define BS(i, j) cutilBankChecker(((float*)&Bs[0][0]), (BLOCK_SIZE * i + 
j))
#else
#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]
#endif

//! Matrix multiplication on the device: C = A * B
//! wA is A's width and wB is B's width
__global__ void
matrixMul( float* C, float* A, float* B, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        AS(ty, tx) = A[a + wA * ty + tx];
        BS(ty, tx) = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();
	    

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += AS(ty, k) * BS(k, tx);

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

__global__ void matrixMul_Restructured(float *C, float *A, float *B, int nIter)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	float Csub[TILE_Y/BLOCK_Y];

	__shared__ float As[TILE_Y][TILE_X];
	__shared__ float Bs[TILE_X][TILE_X];

	for(int l=0; l<nIter; l++){
		for(int j=0; j<TILE_Y/BLOCK_Y; j++)
			Csub[j] = 0;
		for(int i=0; i < Asize/TILE_X; i++){//for all tiles
			for(int j=0; j<TILE_Y; j+=BLOCK_Y)
				As[ty + j][tx] = A[(by * TILE_Y + ty + j) * Asize + i * TILE_X + tx];
			for(int j=0; j<TILE_X; j+=BLOCK_Y)
				Bs[ty + j][tx] = B[(i * TILE_X + ty + j) * Asize + bx * TILE_X + tx];
	
			__syncthreads();

			for(int k=0; k<TILE_Y/BLOCK_Y; k++){
				for(int j=0; j < TILE_X; j++){
					Csub[k] += As[ty + k * BLOCK_Y][j] * Bs[j][tx];
				}
			}
			__syncthreads();
		}

		for(int j=0; j<TILE_Y/BLOCK_Y; j++)
			C[(by * TILE_Y + ty + j * BLOCK_Y) * Asize + bx * TILE_X + tx] = Csub[j];
	}
}

//****************STRASSEN KERNELS********************************************
#ifdef STRASSEN_WITH_GPGPUCOMPILER_KERNELS
__global__ void addition(float * C, float * A, float * B, int widthAB, int widthC, int width)
{
	float sum;
	float a;
	float b;
	int x = blockIdx.x*TILE_X+threadIdx.x;
	int y = blockIdx.y*TILE_Y+threadIdx.y;
	sum=0;
	{
		a=A[(y)*widthAB+(x)];
	}
	{
		b=B[(y)*widthAB+(x)];
	}
	sum=(a+b);
	{
		C[(y)*widthC+(x)]=sum;
	}
}

__global__ void subtraction(float * C, float * A, float * B, int widthA, int widthB, int widthC, int width)
{
	float sum;
	float a;
	float b;
	int x = blockIdx.x*TILE_X+threadIdx.x;
	int y = blockIdx.y*TILE_Y+threadIdx.y;
	sum=0;
	{
		a=A[(y)*widthA+(x)];
	}
	{
		b=B[(y)*widthB+(x)];
	}
	sum=(a-b);
	{
		C[(y)*widthC+(x)]=sum;
	}
}

__global__ void computeC11(float * C, float * m1, float * m4, float * m5, float * m7, int width, int subWidth)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row = by * 16 + ty;
	int col = bx * 16 + tx;
	
	float c[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	#pragma unroll
	for(int i=0; i < 16; i++){
		c[i] = m1[(row+i) * subWidth + col];
		c[i] += m4[(row+i) * subWidth + col];
		c[i] -= m5[(row+i) * subWidth + col];
		c[i] += m7[(row+i) * subWidth + col];
	}
	#pragma unroll
	for(int i=0; i < 16; i++){
		C[(row+i) * width + col] = c[i];
	}
}

__global__ void computeC12(float * C, float * m3, float * m5, int width, int subWidth)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row = by * 16 + ty;
	int col = bx * 16 + tx;
	
	float c[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	#pragma unroll
	for(int i=0; i < 16; i++){
		c[i] = m3[(row+i) * subWidth + col];
		c[i] += m5[(row+i) * subWidth + col];
	}
	#pragma unroll
	for(int i=0; i < 16; i++){
		C[(row+i) * width + col] = c[i];
	}
}

__global__ void computeC21(float * C, float * m2, float * m4, int width, int subWidth)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row = by * 16 + ty;
	int col = bx * 16 + tx;
	
	float c[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	#pragma unroll
	for(int i=0; i < 16; i++){
		c[i] = m2[(row+i) * subWidth + col];
		c[i] += m4[(row+i) * subWidth + col];
	}
	#pragma unroll
	for(int i=0; i < 16; i++){
		C[(row+i) * width + col] = c[i];
	}
}

__global__ void computeC22(float * C, float * m1, float * m2, float * m3, float * m6, int width, int subWidth)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row = by * 16 + ty;
	int col = bx * 16 + tx;
	
	float c[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	#pragma unroll
	for(int i=0; i < 16; i++){
		c[i] = m1[(row+i) * subWidth + col];
		c[i] -= m2[(row+i) * subWidth + col];
		c[i] += m3[(row+i) * subWidth + col];
		c[i] += m6[(row+i) * subWidth + col];
	}
	#pragma unroll
	for(int i=0; i < 16; i++){
		C[(row+i) * width + col] = c[i];
	}
}

#ifdef MULTILEVEL
__global__ void subMatrix(float *B, float *A, int widthB, int widthA)
{
        int tx = threadIdx.x;
        int ty = threadIdx.y;

        int row = blockIdx.y * TILE_Y + ty;
        int column = blockIdx.x * TILE_X + tx;

        __shared__ float as[BLOCK_Y][TILE_X];

        for(int i=0; i < TILE_Y; i+=BLOCK_Y){
		as[ty][tx] = A[(row + i) * widthA + column];
		B[(row + i) * widthB + column] = as[ty][tx];
	}

/*
	int row = blockIdx.y * blockDim.y + ty;
	int column = blockIdx.x * blockDim.x + tx;


	B[row * widthB + column] = A[row * widthA + column];
*/
}
#endif

#else
__global__ void addition (float *C, float *A, float *B, int widthAB, int widthC, int width)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = blockIdx.y * TILE_Y  + ty;
	int column = blockIdx.x * TILE_X + tx;

	__shared__ float as[BLOCK_Y][TILE_X];
	__shared__ float bs[BLOCK_Y][TILE_X];

	#pragma unroll
	for(int i=0; i < TILE_Y; i+=BLOCK_Y){
		as[ty][tx] = A[(row+i)*widthAB+column];
		bs[ty][tx] = B[(row+i)*widthAB+column];
		C[(row + i) * widthC + column] = as[ty][tx] + bs[ty][tx];
	}

/*
	int row = blockIdx.y * blockDim.y + ty;
	int column = blockIdx.x * blockDim.x + tx;

	C[row*widthC+column] = A[row*widthAB+column] + B[row*widthAB+column];
*/
}

__global__ void subtraction (float *C, float *A, float *B, int widthA, int widthB, int widthC, int width)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	int row = blockIdx.y * TILE_Y + ty;
	int column = blockIdx.x * TILE_X + tx;

	__shared__ float as[BLOCK_Y][TILE_X];
	__shared__ float bs[BLOCK_Y][TILE_X];

	#pragma unroll
	for(int i=0; i < TILE_Y; i+=BLOCK_Y){
		as[ty][tx] = A[(row + i) * widthA + column];
		bs[ty][tx] = B[(row + i) * widthB + column];
		C[(row + i) * widthC + column] = as[ty][tx] - bs[ty][tx];
	}
/*
        int row = blockIdx.y * blockDim.y + ty;
        int column = blockIdx.x * blockDim.x + tx;

        C[row*widthC+column] = A[row*widthAB+column] - B[row*widthAB+column];
*/
}

__global__ void multiplication (float *C, float *A, float *B, int widthA, int widthB, int widthC)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;


	int row = blockIdx.y * TILE_Y + ty;
	int column = blockIdx.x * TILE_X + tx;

	float Csub[TILE_Y/BLOCK_Y];

	__shared__ float as[TILE_Y][TILE_X];
	__shared__ float bs[TILE_X][TILE_X];

	for(int i=0; i < TILE_Y/BLOCK_Y; i++)
		Csub[i] = 0;

	for(int i=0; i < widthC; i += TILE_X){
		for(int j=0; j < TILE_Y; j+=BLOCK_Y)
			as[ty + j][tx] = A[(row + j) * widthA + i + tx];
		for(int j=0; j < TILE_X; j+=BLOCK_Y)
			bs[ty + j][tx] = B[(i + ty + j) * widthB + column];

		__syncthreads();

		for(int j=0; j < TILE_Y/BLOCK_Y; j++)
			for(int k=0; k < TILE_X; k++)
				Csub[j] += as[ty+j*BLOCK_Y][k] * bs[k][tx];

		__syncthreads();
	}

	for(int j=0; j < TILE_Y/BLOCK_Y; j++)
		C[(row + j * BLOCK_Y) * widthC + column] = Csub[j];

/*
	int row = blockIdx.y * blockDim.y + ty;
	int column = blockIdx.x * blockDim.x + tx;

	C[row * widthC + column] = 0;

	for(int k=0; k < widthC; k++)
		C[row * widthC + column] += A[row * widthA + k] * B[k * widthB + column];
*/
}

__global__ void computeC11 (float *C, float *m1, float *m4, float *m5, float *m7, int width, int subWidth)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;


	int row = blockIdx.y * TILE_Y + ty;
	int column = blockIdx.x * TILE_X + tx;

	__shared__ float as[BLOCK_Y][TILE_X];

	float Csub;

	#pragma unroll
	for(int i=0; i < TILE_Y; i+=BLOCK_Y){
		as[ty][tx] = m1[(row + i) * subWidth + column];
		Csub = as[ty][tx];
		as[ty][tx] = m4[(row + i) * subWidth + column];
		Csub += as[ty][tx];
		as[ty][tx] = m5[(row + i) * subWidth + column];
		Csub -= as[ty][tx];
		as[ty][tx] = m7[(row + i) * subWidth + column];
		Csub += as[ty][tx];

		C[(row + i) * width + column] = Csub;
	}

/*
	int row = blockIdx.y * blockDim.y + ty;
	int column = blockIdx.x * blockDim.x + tx;

	C[row * width + column] = m1[row * subWidth + column] + m4[row * subWidth + column] - m5[row * subWidth + column] + m7[row * subWidth + column];
*/
}

__global__ void computeC12 (float *C, float *m3, float *m5, int width, int subWidth)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;


	int row = blockIdx.y * TILE_Y + ty;
	int column = blockIdx.x * TILE_X + tx;

	__shared__ float as[BLOCK_Y][TILE_X];
	
	float Csub;

	#pragma unroll
	for(int i=0; i < TILE_Y; i+=BLOCK_Y){
		as[ty][tx] = m3[(row + i) * subWidth + column];
		Csub = as[ty][tx];
		as[ty][tx] = m5[(row + i) * subWidth + column];
		Csub += as[ty][tx];

		C[(row + i) * width + column] = Csub;
	}
/*
        int row = blockIdx.y * blockDim.y + ty;
        int column = blockIdx.x * blockDim.x + tx;

        C[row * width + column] = m3[row * subWidth + column] + m5[row * subWidth + column];
*/
}

__global__ void computeC21 (float *C, float *m2, float *m4, int width, int subWidth)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = blockIdx.y * TILE_Y + ty;
	int column = blockIdx.x * TILE_X + tx;

	__shared__ float as[BLOCK_Y][TILE_X];

	float Csub;

	#pragma unroll
	for(int i=0; i < TILE_Y; i+=BLOCK_Y){
		as[ty][tx] = m2[(row + i) * subWidth + column];
		Csub = as[ty][tx];
		as[ty][tx] = m4[(row + i) * subWidth + column];
		Csub += as[ty][tx];
		
		C[(row + i) * width + column] = Csub;
	}
/*
        int row = blockIdx.y * blockDim.y + ty;
        int column = blockIdx.x * blockDim.x + tx;

        C[row * width + column] = m2[row * subWidth + column] + m4[row * subWidth + column];
*/
}

__global__ void computeC22 (float *C, float *m1, float *m2, float *m3, float *m6, int width,int subWidth)
{
        int tx = threadIdx.x;
        int ty = threadIdx.y;

	int row = blockIdx.y * TILE_Y + threadIdx.y;
	int column = blockIdx.x * TILE_X + threadIdx.x;

        __shared__ float as[BLOCK_Y][TILE_X];

        float Csub;

	#pragma unroll
        for(int i=0; i < TILE_Y; i+=BLOCK_Y){
                as[ty][tx] = m1[(row + i) * subWidth + column];
                Csub = as[ty][tx];
                as[ty][tx] = m2[(row + i) * subWidth + column];
                Csub -= as[ty][tx];
		as[ty][tx] = m3[(row + i) * subWidth + column];
		Csub += as[ty][tx];
		as[ty][tx] = m6[(row + i) * subWidth + column];
		Csub += as[ty][tx];

                C[(row + i) * width + column] = Csub;
        }
/*
        int row = blockIdx.y * blockDim.y + ty;
        int column = blockIdx.x * blockDim.x + tx;

        C[row * width + column] = m1[row * subWidth + column] - m2[row * subWidth + column] + m3[row * subWidth + column] + m6[row * subWidth + column];
*/
}


#ifdef MULTILEVEL
__global__ void subMatrix(float *B, float *A, int widthB, int widthA)
{
        int tx = threadIdx.x;
        int ty = threadIdx.y;

        int row = blockIdx.y * TILE_Y + ty;
        int column = blockIdx.x * TILE_X + tx;

        __shared__ float as[BLOCK_Y][TILE_X];

	#pragma unroll
        for(int i=0; i < TILE_Y; i+=BLOCK_Y){
		as[ty][tx] = A[(row + i) * widthA + column];
		B[(row + i) * widthB + column] = as[ty][tx];
	}

/*
	int row = blockIdx.y * blockDim.y + ty;
	int column = blockIdx.x * blockDim.x + tx;

	B[row * widthB + column] = A[row * widthA + column];
*/
}
#endif
#endif
//************END OF STRASSEN'S KERNELS Our Approach*************

//*********START OF STRASSEN'S KERNELS PAPER VERSION************

__device__ void update2(float *a, float b, float *c)
{
	for(int i=0; i < 16; i++)
		c[i] += a[i*4] * b;
}

__global__ void GPU8(float *a, float *b, float *c, int n)
{
	__shared__ float as[16][65];

	float cr[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

	int nDiv64 = n/64;
	int sRow = threadIdx.y;
	int sRow4 = sRow * 4;
	int sCol = threadIdx.x;
	int tid = sRow * 16 + sCol;
	int aNext = (16 * blockIdx.y + sRow) * n + sCol * 4;
	int bNext = 128 * blockIdx.x + tid;
	int cNext = 16 * blockIdx.y * n + 128 * blockIdx.x + tid;
	int nTimes2 = 2 * n;
	int nTimes3 = 3 * n;
	int nTimes4 = 4 * n;

	a += aNext;
	b += bNext;
	c += cNext;

	float4 *a4 = (float4 *)a;

	for(int i=0; i < nDiv64; i++){
		*((float4 *)(&as[sCol][sRow4])) = a4[0];
		*((float4 *)(&as[sCol][sRow4+32])) = a4[nTimes2];
		__syncthreads();

		float br0 = b[0];
		float br1 = b[n];
		float br2 = b[nTimes2];
		float br3 = b[nTimes3];

		b += nTimes4;

		#pragma unroll
		for(int k=0; k < 15; k++){
			update2(&as[k][0], br0, cr); br0 = b[0];
			update2(&as[k][1], br1, cr); br1 = b[n];
			update2(&as[k][2], br2, cr); br2 = b[nTimes2];
			update2(&as[k][3], br3, cr); br3 = b[nTimes3];

			b += nTimes4;
		}

		update2(&as[15][0], br0, cr);
		update2(&as[15][1], br1, cr);
		update2(&as[15][2], br2, cr);
		update2(&as[15][3], br3, cr);

		a4 += 16;
		__syncthreads();
	}

	for(int j=0; j < 16; j++){
		c[0] = cr[j];
		c += n;
	}
}

__global__ void strassen_GPU8(float *a, float *b, float *c, int W, int widthA, int widthB, int widthC)
{
        __shared__ float as[16][65];

        float cr[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

        int nDiv64 = W/64;
        int sRow = threadIdx.y;
        int sRow4 = sRow * 4;
        int sCol = threadIdx.x;
        int tid = sRow * 16 + sCol;
        int aNext = (16 * blockIdx.y + sRow) * widthA + sCol * 4;
        int bNext = 128 * blockIdx.x + tid;
        int cNext = 16 * blockIdx.y * widthC + 128 * blockIdx.x + tid;
	int nTimes2a = 2 * widthA;
	int n = widthB;
        int nTimes2 = 2 * n;
        int nTimes3 = 3 * n;
        int nTimes4 = 4 * n;

        a += aNext;
        b += bNext;
        c += cNext;

        float4 *a4 = (float4 *)a;

        for(int i=0; i < nDiv64; i++){
                *((float4 *)(&as[sCol][sRow4])) = a4[0];
                *((float4 *)(&as[sCol][sRow4+32])) = a4[nTimes2a];
                __syncthreads();

                float br0 = b[0];
                float br1 = b[n];
                float br2 = b[nTimes2];
                float br3 = b[nTimes3];

                b += nTimes4;

                #pragma unroll
                for(int k=0; k < 15; k++){
                        update2(&as[k][0], br0, cr); br0 = b[0];
                        update2(&as[k][1], br1, cr); br1 = b[n];
                        update2(&as[k][2], br2, cr); br2 = b[nTimes2];
                        update2(&as[k][3], br3, cr); br3 = b[nTimes3];

                        b += nTimes4;
                }

                update2(&as[15][0], br0, cr);
                update2(&as[15][1], br1, cr);
                update2(&as[15][2], br2, cr);
                update2(&as[15][3], br3, cr);

                a4 += 16;
                __syncthreads();
        }

        for(int j=0; j < 16; j++){
                c[0] = cr[j];
                c += widthC;
        }
}


__global__ void add(float *d_A, float *d_B, float *d_C, int widthA, int widthB, int widthC)
{
	int startA = blockIdx.x * 64 + threadIdx.x * 2 + (blockIdx.y * 8 + threadIdx.y) * widthA;
        int startB = blockIdx.x * 64 + threadIdx.x * 2 + (blockIdx.y * 8 + threadIdx.y) * widthB;
        int startC = blockIdx.x * 64 + threadIdx.x * 2 + (blockIdx.y * 8 + threadIdx.y) * widthC;

	float2 tempA = *(float2 *)(d_A+startA);
	float2 tempB = *(float2 *)(d_B+startB);

	tempA.x += tempB.x;
	tempA.y += tempB.y;

	*(float2 *)(d_C+startC) = tempA;
}

__global__ void sub(float *d_A, float *d_B, float *d_C, int widthA, int widthB, int widthC)
{
        int startA = blockIdx.x * 64 + threadIdx.x * 2 + (blockIdx.y * 8 + threadIdx.y) * widthA;
        int startB = blockIdx.x * 64 + threadIdx.x * 2 + (blockIdx.y * 8 + threadIdx.y) * widthB;
        int startC = blockIdx.x * 64 + threadIdx.x * 2 + (blockIdx.y * 8 + threadIdx.y) * widthC;

        float2 tempA = *(float2 *)(d_A+startA);
        float2 tempB = *(float2 *)(d_B+startB);

        tempA.x -= tempB.x;
        tempA.y -= tempB.y;

        *(float2 *)(d_C+startC) = tempA;
}

__global__ void add_add(float *d_A, float *d_B, float *d_C, int widthA, int widthB, int widthC)
{
        int startA = blockIdx.x * 64 + threadIdx.x * 2 + (blockIdx.y * 8 + threadIdx.y) * widthA;
        int startB = blockIdx.x * 64 + threadIdx.x * 2 + (blockIdx.y * 8 + threadIdx.y) * widthB;
        int startC = blockIdx.x * 64 + threadIdx.x * 2 + (blockIdx.y * 8 + threadIdx.y) * widthC;

	float2 tempA = *(float2 *)(d_A+startA);
        float2 tempB = *(float2 *)(d_A+startB);
        float2 tempC = *(float2 *)(d_B+startC);

        tempB.x += tempA.x;
        tempB.y += tempA.y;

        tempC.x += tempA.x;
        tempC.y += tempA.y;

        *(float2 *)(d_B+startB) = tempB;
        *(float2 *)(d_C+startC) = tempC;
}

__global__ void add_sub(float *d_A, float *d_B, float *d_C, int widthA, int widthB, int widthC)
{
        int startA = blockIdx.x * 64 + threadIdx.x * 2 + (blockIdx.y * 8 + threadIdx.y) * widthA;
        int startB = blockIdx.x * 64 + threadIdx.x * 2 + (blockIdx.y * 8 + threadIdx.y) * widthB;
        int startC = blockIdx.x * 64 + threadIdx.x * 2 + (blockIdx.y * 8 + threadIdx.y) * widthC;

        float2 tempA = *(float2 *)(d_A+startA);
        float2 tempB = *(float2 *)(d_A+startB);
        float2 tempC = *(float2 *)(d_B+startC);

        tempB.x += tempA.x;
        tempB.y += tempA.y;

        tempC.x -= tempA.x;
        tempC.y -= tempA.y;

        *(float2 *)(d_B+startB) = tempB;
        *(float2 *)(d_C+startC) = tempC;
}


__global__ void mulIncInc(float *a, float *b, float *c, float *d, int W, int widthA, int widthB, int widthC, int widthD)
{
        __shared__ float as[16][65];

        float cr[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

        int nDiv64 = W/64;
        int sRow = threadIdx.y;
        int sRow4 = sRow * 4;
        int sCol = threadIdx.x;
        int tid = sRow * 16 + sCol;
        int aNext = (16 * blockIdx.y + sRow) * widthA + sCol * 4;
        int bNext = 128 * blockIdx.x + tid;
        int cNext = 16 * blockIdx.y * widthC + 128 * blockIdx.x + tid;
	int dNext = 16 * blockIdx.y * widthD + 128 * blockIdx.x + tid;
	int nTimes2a = 2 * widthA;
	int n = widthB;
        int nTimes2 = 2 * n;
        int nTimes3 = 3 * n;
        int nTimes4 = 4 * n;

        a += aNext;
        b += bNext;
        c += cNext;
	d += dNext;

        float4 *a4 = (float4 *)a;

        for(int i=0; i < nDiv64; i++){
                *((float4 *)(&as[sCol][sRow4])) = a4[0];
                *((float4 *)(&as[sCol][sRow4+32])) = a4[nTimes2a];
                __syncthreads();

                float br0 = b[0];
                float br1 = b[n];
                float br2 = b[nTimes2];
                float br3 = b[nTimes3];

                b += nTimes4;

                #pragma unroll
                for(int k=0; k < 15; k++){
                        update2(&as[k][0], br0, cr); br0 = b[0];
                        update2(&as[k][1], br1, cr); br1 = b[n];
                        update2(&as[k][2], br2, cr); br2 = b[nTimes2];
                        update2(&as[k][3], br3, cr); br3 = b[nTimes3];

                        b += nTimes4;
                }

                update2(&as[15][0], br0, cr);
                update2(&as[15][1], br1, cr);
                update2(&as[15][2], br2, cr);
                update2(&as[15][3], br3, cr);

                a4 += 16;
                __syncthreads();
        }

        for(int j=0; j < 16; j++){
                c[0] += cr[j];
		d[0] += cr[j];
                c += widthC;
		d += widthD;
        }
}

__global__ void mulIncDec(float *a, float *b, float *c, float *d, int W, int widthA, int widthB, int widthC, int widthD)
{
        __shared__ float as[16][65];

        float cr[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

        int nDiv64 = W/64;
        int sRow = threadIdx.y;
        int sRow4 = sRow * 4;
        int sCol = threadIdx.x;
        int tid = sRow * 16 + sCol;
        int aNext = (16 * blockIdx.y + sRow) * widthA + sCol * 4;
        int bNext = 128 * blockIdx.x + tid;
        int cNext = 16 * blockIdx.y * widthC + 128 * blockIdx.x + tid;
        int dNext = 16 * blockIdx.y * widthD + 128 * blockIdx.x + tid;
	int nTimes2a = 2 * widthA;
	int n = widthB;
        int nTimes2 = 2 * n;
        int nTimes3 = 3 * n;
        int nTimes4 = 4 * n;

        a += aNext;
        b += bNext;
        c += cNext;
        d += dNext;

        float4 *a4 = (float4 *)a;

        for(int i=0; i < nDiv64; i++){
                *((float4 *)(&as[sCol][sRow4])) = a4[0];
                *((float4 *)(&as[sCol][sRow4+32])) = a4[nTimes2a];
                __syncthreads();

                float br0 = b[0];
                float br1 = b[n];
                float br2 = b[nTimes2];
                float br3 = b[nTimes3];

                b += nTimes4;

                #pragma unroll
                for(int k=0; k < 15; k++){
                        update2(&as[k][0], br0, cr); br0 = b[0];
                        update2(&as[k][1], br1, cr); br1 = b[n];
                        update2(&as[k][2], br2, cr); br2 = b[nTimes2];
                        update2(&as[k][3], br3, cr); br3 = b[nTimes3];

                        b += nTimes4;
                }

                update2(&as[15][0], br0, cr);
                update2(&as[15][1], br1, cr);
                update2(&as[15][2], br2, cr);
                update2(&as[15][3], br3, cr);

                a4 += 16;
                __syncthreads();
        }

        for(int j=0; j < 16; j++){
                c[0] += cr[j];
                d[0] -= cr[j];
                c += widthC;
                d += widthD;
        }
}

__global__ void mulStoreDec(float *a, float *b, float *c, float *d, int W, int widthA, int widthB, int widthC, int widthD)
{
        __shared__ float as[16][65];

        float cr[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

        int nDiv64 = W/64;
        int sRow = threadIdx.y;
        int sRow4 = sRow * 4;
        int sCol = threadIdx.x;
        int tid = sRow * 16 + sCol;
        int aNext = (16 * blockIdx.y + sRow) * widthA + sCol * 4;
        int bNext = 128 * blockIdx.x + tid;
        int cNext = 16 * blockIdx.y * widthC + 128 * blockIdx.x + tid;
        int dNext = 16 * blockIdx.y * widthD + 128 * blockIdx.x + tid;
	int nTimes2a = 2 * widthA;
	int n = widthB;
        int nTimes2 = 2 * n;
        int nTimes3 = 3 * n;
        int nTimes4 = 4 * n;

        a += aNext;
        b += bNext;
        c += cNext;
        d += dNext;

        float4 *a4 = (float4 *)a;

        for(int i=0; i < nDiv64; i++){
                *((float4 *)(&as[sCol][sRow4])) = a4[0];
                *((float4 *)(&as[sCol][sRow4+32])) = a4[nTimes2a];
                __syncthreads();

                float br0 = b[0];
                float br1 = b[n];
                float br2 = b[nTimes2];
                float br3 = b[nTimes3];

                b += nTimes4;

                #pragma unroll
                for(int k=0; k < 15; k++){
                        update2(&as[k][0], br0, cr); br0 = b[0];
                        update2(&as[k][1], br1, cr); br1 = b[n];
                        update2(&as[k][2], br2, cr); br2 = b[nTimes2];
                        update2(&as[k][3], br3, cr); br3 = b[nTimes3];

                        b += nTimes4;
                }

                update2(&as[15][0], br0, cr);
                update2(&as[15][1], br1, cr);
                update2(&as[15][2], br2, cr);
                update2(&as[15][3], br3, cr);

                a4 += 16;
                __syncthreads();
        }

        for(int j=0; j < 16; j++){
                c[0] = cr[j];
                d[0] -= cr[j];
                c += widthC;
                d += widthD;
        }
}

__global__ void mulStoreInc(float *a, float *b, float *c, float *d, int W, int widthA, int widthB, int widthC, int widthD)
{
        __shared__ float as[16][65];

        float cr[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

        int nDiv64 = W/64;
        int sRow = threadIdx.y;
        int sRow4 = sRow * 4;
        int sCol = threadIdx.x;
        int tid = sRow * 16 + sCol;
        int aNext = (16 * blockIdx.y + sRow) * widthA + sCol * 4;
        int bNext = 128 * blockIdx.x + tid;
        int cNext = 16 * blockIdx.y * widthC + 128 * blockIdx.x + tid;
        int dNext = 16 * blockIdx.y * widthD + 128 * blockIdx.x + tid;
	int nTimes2a = 2 * widthA;
	int n = widthB;
        int nTimes2 = 2 * n;
        int nTimes3 = 3 * n;
        int nTimes4 = 4 * n;

        a += aNext;
        b += bNext;
        c += cNext;
        d += dNext;

        float4 *a4 = (float4 *)a;

        for(int i=0; i < nDiv64; i++){
                *((float4 *)(&as[sCol][sRow4])) = a4[0];
                *((float4 *)(&as[sCol][sRow4+32])) = a4[nTimes2a];
                __syncthreads();

                float br0 = b[0];
                float br1 = b[n];
                float br2 = b[nTimes2];
                float br3 = b[nTimes3];

                b += nTimes4;

                #pragma unroll
                for(int k=0; k < 15; k++){
                        update2(&as[k][0], br0, cr); br0 = b[0];
                        update2(&as[k][1], br1, cr); br1 = b[n];
                        update2(&as[k][2], br2, cr); br2 = b[nTimes2];
                        update2(&as[k][3], br3, cr); br3 = b[nTimes3];

                        b += nTimes4;
                }

                update2(&as[15][0], br0, cr);
                update2(&as[15][1], br1, cr);
                update2(&as[15][2], br2, cr);
                update2(&as[15][3], br3, cr);

                a4 += 16;
                __syncthreads();
        }

        for(int j=0; j < 16; j++){
                c[0] = cr[j];
                d[0] += cr[j];
                c += widthC;
                d += widthD;
        }
}

__global__ void mulAdd(float *a, float *b, float *c, float *d, int W, int widthA, int widthB, int widthC, int widthD)
{
        __shared__ float as[16][65];

        float cr[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

        int nDiv64 = W/64;
        int sRow = threadIdx.y;
        int sRow4 = sRow * 4;
        int sCol = threadIdx.x;
        int tid = sRow * 16 + sCol;
        int aNext = (16 * blockIdx.y + sRow) * widthA + sCol * 4;
        int bNext = 128 * blockIdx.x + tid;
        int cNext = 16 * blockIdx.y * widthC + 128 * blockIdx.x + tid;
        int dNext = 16 * blockIdx.y * widthD + 128 * blockIdx.x + tid;
	int nTimes2a = 2 * widthA;
	int n = widthB;
        int nTimes2 = 2 * n;
        int nTimes3 = 3 * n;
        int nTimes4 = 4 * n;

        a += aNext;
        b += bNext;
        c += cNext;
        d += dNext;

       float4 *a4 = (float4 *)a;

        for(int i=0; i < nDiv64; i++){
                *((float4 *)(&as[sCol][sRow4])) = a4[0];
                *((float4 *)(&as[sCol][sRow4+32])) = a4[nTimes2a];
                __syncthreads();

                float br0 = b[0];
                float br1 = b[n];
                float br2 = b[nTimes2];
                float br3 = b[nTimes3];

                b += nTimes4;

                #pragma unroll
                for(int k=0; k < 15; k++){
                        update2(&as[k][0], br0, cr); br0 = b[0];
                        update2(&as[k][1], br1, cr); br1 = b[n];
                        update2(&as[k][2], br2, cr); br2 = b[nTimes2];
                        update2(&as[k][3], br3, cr); br3 = b[nTimes3];

                        b += nTimes4;
                }

                update2(&as[15][0], br0, cr);
                update2(&as[15][1], br1, cr);
                update2(&as[15][2], br2, cr);
                update2(&as[15][3], br3, cr);

                a4 += 16;
                __syncthreads();
        }

        for(int j=0; j < 16; j++){
                d[0] = cr[j] + c[0];
                c += widthC;
                d += widthD;
        }
}

__global__ void mulIncIncInc(float *a, float *b, float *c, float *d, float *e, float *f, int W, int widthA, int widthB, int widthC, int widthD, int widthE, int widthF)
{
        __shared__ float as[16][65];

        float cr[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

        int nDiv64 = W/64;
        int sRow = threadIdx.y;
        int sRow4 = sRow * 4;
        int sCol = threadIdx.x;
        int tid = sRow * 16 + sCol;
        int aNext = (16 * blockIdx.y + sRow) * widthA + sCol * 4;
        int bNext = 128 * blockIdx.x + tid;
        int cNext = 16 * blockIdx.y * widthC + 128 * blockIdx.x + tid;
        int dNext = 16 * blockIdx.y * widthD + 128 * blockIdx.x + tid;
        int eNext = 16 * blockIdx.y * widthE + 128 * blockIdx.x + tid;
        int fNext = 16 * blockIdx.y * widthF + 128 * blockIdx.x + tid;
	int nTimes2a = 2 * widthA;
	int n = widthB;
        int nTimes2 = 2 * n;
        int nTimes3 = 3 * n;
        int nTimes4 = 4 * n;

        a += aNext;
        b += bNext;
        c += cNext;
        d += dNext;
	e += eNext;
	f += fNext;

       float4 *a4 = (float4 *)a;

        for(int i=0; i < nDiv64; i++){
                *((float4 *)(&as[sCol][sRow4])) = a4[0];
                *((float4 *)(&as[sCol][sRow4+32])) = a4[nTimes2a];
                __syncthreads();

                float br0 = b[0];
                float br1 = b[n];
                float br2 = b[nTimes2];
                float br3 = b[nTimes3];

                b += nTimes4;

                #pragma unroll
                for(int k=0; k < 15; k++){
                        update2(&as[k][0], br0, cr); br0 = b[0];
                        update2(&as[k][1], br1, cr); br1 = b[n];
                        update2(&as[k][2], br2, cr); br2 = b[nTimes2];
                        update2(&as[k][3], br3, cr); br3 = b[nTimes3];

                        b += nTimes4;
                }

                update2(&as[15][0], br0, cr);
                update2(&as[15][1], br1, cr);
                update2(&as[15][2], br2, cr);
                update2(&as[15][3], br3, cr);

                a4 += 16;
                __syncthreads();
        }

        for(int j=0; j < 16; j++){
		c[0] = cr[j];
		float temp_e = e[0] + cr[j];
		f[0] += temp_e;
		e[0] = temp_e + d[0];
                c += widthC;
                d += widthD;
		e += widthE;
		f += widthF;
        }
}

__global__ void mulSubInc(float *a, float *b, float *c, float *d, float *e, int W, int widthA, int widthB, int widthC, int widthD, int widthE)
{
        __shared__ float as[16][65];

        float cr[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

        int nDiv64 = W/64;
        int sRow = threadIdx.y;
        int sRow4 = sRow * 4;
        int sCol = threadIdx.x;
        int tid = sRow * 16 + sCol;
        int aNext = (16 * blockIdx.y + sRow) * widthA + sCol * 4;
        int bNext = 128 * blockIdx.x + tid;
        int cNext = 16 * blockIdx.y * widthC + 128 * blockIdx.x + tid;
        int dNext = 16 * blockIdx.y * widthD + 128 * blockIdx.x + tid;
        int eNext = 16 * blockIdx.y * widthE + 128 * blockIdx.x + tid;
	int nTimes2a = 2 * widthA;
	int n = widthB;
        int nTimes2 = 2 * n;
        int nTimes3 = 3 * n;
        int nTimes4 = 4 * n;

        a += aNext;
        b += bNext;
        c += cNext;
        d += dNext;
        e += eNext;

       float4 *a4 = (float4 *)a;

        for(int i=0; i < nDiv64; i++){
                *((float4 *)(&as[sCol][sRow4])) = a4[0];
                *((float4 *)(&as[sCol][sRow4+32])) = a4[nTimes2a];
                __syncthreads();

                float br0 = b[0];
                float br1 = b[n];
                float br2 = b[nTimes2];
                float br3 = b[nTimes3];

                b += nTimes4;

                #pragma unroll
                for(int k=0; k < 15; k++){
                        update2(&as[k][0], br0, cr); br0 = b[0];
                        update2(&as[k][1], br1, cr); br1 = b[n];
                        update2(&as[k][2], br2, cr); br2 = b[nTimes2];
                        update2(&as[k][3], br3, cr); br3 = b[nTimes3];

                        b += nTimes4;
                }

                update2(&as[15][0], br0, cr);
                update2(&as[15][1], br1, cr);
                update2(&as[15][2], br2, cr);
                update2(&as[15][3], br3, cr);

                a4 += 16;
                __syncthreads();
        }

        for(int j=0; j < 16; j++){
		float temp_c = c[0];
		d[0] = temp_c - cr[j];
		e[0] += temp_c;
                c += widthC;
                d += widthD;
                e += widthE;
        }
}

//********************END OF STRASSEN KERNELS PAPER VERSION*************************

#endif // #ifndef _MATRIXMUL_KERNEL_H_

