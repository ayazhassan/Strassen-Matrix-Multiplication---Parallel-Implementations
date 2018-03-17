
#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixMul.h"

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
__global__ void addition (float *C, float *A, float *B, int widthAB, int widthC, int width)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = blockIdx.y * blockDim.y  + ty;
	int column = blockIdx.x * blockDim.x + tx;

	float Csub[TILE_Y/BLOCK_Y];

	__shared__ float as[TILE_Y][TILE_X];
	__shared__ float bs[TILE_Y][TILE_X];

	for(int i=0; i < TILE_Y/blockDim.y; i++){
		as[ty+i*blockDim.y][tx] = A[(row+i*blockDim.y)*widthAB+column];
		bs[ty+i*blockDim.y][tx] = B[(row+i*blockDim.y)*widthAB+column];
//		C[(row + i*blockDim.y) * widthC + column] = as[ty+i*blockDim.y][tx] + bs[ty+i*blockDim.y][tx];
	}

	for(int i=0; i < TILE_Y/blockDim.y; i++){
		Csub[i] = as[ty+i*blockDim.y][tx] + bs[ty+i*blockDim.y][tx];
		C[(row + i*blockDim.y) * widthC + column] = Csub[i];
	}

}

__global__ void subtraction (float *C, float *A, float *B, int widthAB, int widthC, int width)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int column = blockIdx.x * blockDim.x + threadIdx.x;

	C[row*widthC+column] = A[row*widthAB+column] - B[row*widthAB+column];
}

__global__ void multiplication (float *C, float *A, float *B, int widthA, int widthB, int widthC)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int column = blockIdx.x * blockDim.x + threadIdx.x;

	C[row*widthC+column] = 0;

	for (int k=0; k< widthC; k++)
		C[row*widthC+column] += A[row*widthA+k]*B[k*widthB+column];
}

__global__ void computeC11 (float *C, float *m1, float *m4, float *m5, float *m7, int width, int subWidth)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int column = blockIdx.x * blockDim.x + threadIdx.x;

	C[row*width+column] = m1[row*subWidth+column] + m4[row*subWidth+column] - m5[row*subWidth+column] + m7[row*subWidth+column];
}

__global__ void computeC12 (float *C, float *m3, float *m5, int width, int subWidth)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int column = blockIdx.x * blockDim.x + threadIdx.x;

	C[row*width+column] = m3[row*subWidth+column] + m5[row*subWidth+column];
}

__global__ void computeC21 (float *C, float *m2, float *m4, int width, int subWidth)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int column = blockIdx.x * blockDim.x + threadIdx.x;

	C[row*width+column] = m2[row*subWidth+column] + m4[row*subWidth+column];
}

__global__ void computeC22 (float *C, float *m1, float *m2, float *m3, float *m6, int width,int subWidth)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int column = blockIdx.x * blockDim.x + threadIdx.x;

	C[row*width+column] = m1[row*subWidth+column] - m2[row*subWidth+column] + m3[row*subWidth+column] + m6[row*subWidth+column];
}


#ifdef MULTILEVEL
__global__ void subMatrix(float *B, float *A, int widthB, int widthA)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int column = blockIdx.x * blockDim.x + threadIdx.x;

	B[row * widthB + column] = A[row * widthA + column];
}
#endif
//********************END OF STRASSEN KERNELS*************************

#endif // #ifndef _MATRIXMUL_KERNEL_H_

