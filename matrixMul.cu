// Utilities and system includes
#include <cublas_v2.h>
#include "cutil_inline.h"
#include <sdkHelper.h>  // helper for shared functions common to CUDA SDK samples
#include <shrQATest.h>
#include <shrUtils.h>
#include <cuda_runtime.h>
#define shrLog printf
#define shrLogEx printf

// includes, kernels
#include "matrixMul_kernel.h"

static char *sSDKsample = "StrassenmatrixMul";

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions

void inline checkError(cublasStatus_t status, const char* msg)
{
    if(status != CUBLAS_STATUS_SUCCESS){
        printf("%s", msg);
        exit(-1);
    }
}
    // This will output the proper CUDA error strings in the event that a CUDA host call returns an error
    #define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)

    inline void __checkCudaErrors( cudaError err, const char *file, const int line )
    {
        if( cudaSuccess != err) {
		    fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                    file, line, (int)err, cudaGetErrorString( err ) );
            exit(-1);
        }
    }

    // This will output the proper error string when calling cudaGetLastError
    #define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

    inline void __getLastCudaError( const char *errorMessage, const char *file, const int line )
    {
        cudaError_t err = cudaGetLastError();
        if( cudaSuccess != err) {
            fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
                    file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
            exit(-1);
        }
    }

    // General GPU Device CUDA Initialization
    int gpuDeviceInit(int devID)
    {
        int deviceCount;
        checkCudaErrors(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) {
            fprintf(stderr, "gpuDeviceInit() CUDA error: no devices supporting CUDA.\n");
            exit(-1);
        }
        if (devID < 0) 
            devID = 0;
        if (devID > deviceCount-1) {
            fprintf(stderr, "\n");
            fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", deviceCount);
            fprintf(stderr, ">> gpuDeviceInit (-device=%d) is not a valid GPU device. <<\n", devID);
            fprintf(stderr, "\n");
            return -devID;
        }

        cudaDeviceProp deviceProp;
        checkCudaErrors( cudaGetDeviceProperties(&deviceProp, devID) );
        if (deviceProp.major < 1) {
            fprintf(stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n");
            exit(-1);                                                  \
        }

        checkCudaErrors( cudaSetDevice(devID) );
        printf("> gpuDeviceInit() CUDA device [%d]: %s\n", devID, deviceProp.name);
        return devID;
    }

    // This function returns the best GPU (with maximum GFLOPS)
    int gpuGetMaxGflopsDeviceId()
    {
	    int current_device   = 0, sm_per_multiproc = 0;
	    int max_compute_perf = 0, max_perf_device  = 0;
	    int device_count     = 0, best_SM_arch     = 0;
	    cudaDeviceProp deviceProp;

	    cudaGetDeviceCount( &device_count );
	    // Find the best major SM Architecture GPU device
	    while ( current_device < device_count ) {
		    cudaGetDeviceProperties( &deviceProp, current_device );
		    if (deviceProp.major > 0 && deviceProp.major < 9999) {
			    best_SM_arch = MAX(best_SM_arch, deviceProp.major);
		    }
		    current_device++;
	    }

        // Find the best CUDA capable GPU device
        current_device = 0;
        while( current_device < device_count ) {
           cudaGetDeviceProperties( &deviceProp, current_device );
           if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
               sm_per_multiproc = 1;
		   } else {
               sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
           }

           int compute_perf  = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;
           if( compute_perf  > max_compute_perf ) {
               // If we find GPU with SM major > 2, search only these
               if ( best_SM_arch > 2 ) {
                   // If our device==dest_SM_arch, choose this, or else pass
                   if (deviceProp.major == best_SM_arch) {	
                       max_compute_perf  = compute_perf;
                       max_perf_device   = current_device;
                   }
               } else {
                   max_compute_perf  = compute_perf;
                   max_perf_device   = current_device;
               }
           }
           ++current_device;
	    }
	    return max_perf_device;
    }

    // Initialization code to find the best CUDA Device
    int findCudaDevice(int argc, const char **argv)
    {
        cudaDeviceProp deviceProp;
        int devID = 0;
        // If the command-line has a device number specified, use it
        if (checkCmdLineFlag(argc, argv, "device")) {
            devID = getCmdLineArgumentInt(argc, argv, "device=");
            if (devID < 0) {
                printf("Invalid command line parameters\n");
                exit(-1);
            } else {
                devID = gpuDeviceInit(devID);
                if (devID < 0) {
                   printf("exiting...\n");
                   shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
                   exit(-1);
                }
            }
        } else {
            // Otherwise pick the device with highest Gflops/s
            devID = gpuGetMaxGflopsDeviceId();
            checkCudaErrors( cudaSetDevice( devID ) );
            checkCudaErrors( cudaGetDeviceProperties(&deviceProp, devID) );
            printf("> Using CUDA device [%d]: %s\n", devID, deviceProp.name);
        }
        return devID;
    }
// end of CUDA Helper Functions



////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char** argv);
void randomInit(float*, int);
void printDiff(float*, float*, int, int, int, float);

void
computeGold(float* C, const float* A, const float* B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j) {
            double sum = 0;
            for (unsigned int k = 0; k < wA; ++k) {
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum += a * b;
            }
            C[i * wB + j] = (float)sum;
        }
}

int check_result(float *a, float *b, int size)
{
	float temp;
        for(int i=0; i<size; i++){
		temp = a[i] - b[i];
                if(temp > 0.1 || temp < -0.1 ){
                        printf("index: (%d, %d), correct = %f, incorrect = %f\n", i/(size/N), i % (size/N), a[i], b[i]);
                        return 1;
                }
	}
        return 0;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
	printf("[ %s ]\n", sSDKsample);

    shrLog("%s Starting...\n\n", argv[0]);

    runTest(argc, argv);

    return 0;
}

void strassen_paper_base(float *A, float *B, float *C, int width, int widthA, int widthB, int widthC)
{
	float *t1, *t2;

	int subWidth = width/2;
	int memSize = subWidth * subWidth * sizeof(float);

	int shiftX = subWidth;
	int shiftY = width * subWidth;

	cudaMalloc((void**) &t1, memSize);
	cudaMalloc((void**) &t2, memSize);

	dim3 thread_mul(16, 8);
	dim3 grid_mul(subWidth/128, subWidth/16);

	dim3 thread_add(32, 8);
	dim3 grid_add(subWidth/64, subWidth/8);

	sub <<< grid_add, thread_add >>> (A+shiftY, A, C+shiftX, widthA, widthA, widthC);
	add <<< grid_add, thread_add >>> (B, B+shiftX, C+shiftY, widthB, widthB, widthC);
	strassen_GPU8 <<< grid_mul, thread_mul >>> (C+shiftX, C+shiftY, C+shiftX+shiftY, subWidth, widthC, widthC, widthC);
	sub <<< grid_add, thread_add >>> (A+shiftX, A+shiftX+shiftY, C+shiftX, widthA, widthA, widthC);
	add <<< grid_add, thread_add >>> (B+shiftY, B+shiftX+shiftY, C+shiftY, widthB, widthB, widthC);
	strassen_GPU8 <<< grid_mul, thread_mul >>> (C+shiftX, C+shiftY, C, subWidth, widthC, widthC, widthC);
	add <<< grid_add, thread_add >>> (A, A+shiftX+shiftY, C+shiftX, widthA, widthA, widthC);
	add <<< grid_add, thread_add >>> (B, B+shiftX+shiftY, C+shiftY, widthB, widthB, widthC);
	mulIncInc <<< grid_mul, thread_mul >>> (C+shiftX, C+shiftY, C, C+shiftX+shiftY, subWidth, widthC, widthC, widthC, widthC);
	add <<< grid_add, thread_add >>> (A+shiftY, A+shiftX+shiftY, t2, widthA, widthA, subWidth);
	mulStoreDec <<< grid_mul, thread_mul >>> (t2, B, C+shiftY, C+shiftX+shiftY, subWidth, subWidth, widthB, widthC, widthC);
	sub <<< grid_add, thread_add >>> (B+shiftY, B, t1, widthB, widthB, subWidth);
	mulIncInc <<< grid_mul, thread_mul >>> (A+shiftX+shiftY, t1, C+shiftY, C, subWidth, widthA, subWidth, widthC, widthC);
	sub <<< grid_add, thread_add >>> (B+shiftX, B+shiftX+shiftY, t1, widthB, widthB, subWidth);
	mulStoreInc <<< grid_mul, thread_mul >>> (A, t1, C+shiftX, C+shiftX+shiftY, subWidth, widthA, subWidth, widthC, widthC);
	add <<< grid_add, thread_add >>> (A, A+shiftX, t2, widthA, widthA, subWidth);
	mulIncDec <<< grid_mul, thread_mul >>> (t2, B+shiftX+shiftY, C+shiftX, C, subWidth, subWidth, widthB, widthC, widthC);

	cudaFree(t1);
	cudaFree(t2);
	
}

void strassen_paper_recursive(float *A, float *B, float *C, int width, int widthA, int widthB, int widthC)
{
	if(width <= TAU1){
//		printf("Running strassen_GPU8 with width = %d\n", width);
		dim3 threads(16, 8);
		dim3 grid(width/128, width/16);

		strassen_GPU8 <<< grid, threads >>> (A, B, C, width, widthA, widthB, widthC);
	}
	else if(width <= TAU2){
//		printf("Running strassen_paper_base with width = %d\n", width);
		strassen_paper_base(A, B, C, width, widthA, widthB, widthC);

	}
	else{
//		printf("Running strassen_paper_recursive with width = %d\n", width);
		int subWidth = width / 2;
		int memSize = subWidth * subWidth * sizeof(float);

		float *t1, *t2;
		cudaMalloc((void **)&t1, memSize);
                cudaMalloc((void **)&t2, memSize);
		
		int shiftX = subWidth;
		int shiftY = width * subWidth;
		
		dim3 thread_add(32, 8);
		dim3 grid_add(subWidth/64, subWidth/8);

		dim3 thread_mul(16, 8);
		dim3 grid_mul(subWidth/128, subWidth/16);

		sub <<< grid_add, thread_add >>> (A+shiftY, A, C+shiftX, widthA, widthA, widthC);
		add <<< grid_add, thread_add >>> (B, B+shiftX, C+shiftY, widthB, widthB, widthC);

		strassen_paper_recursive(C+shiftX, C+shiftY, C+shiftX+shiftY, subWidth, widthC, widthC, widthC);

		sub <<< grid_add, thread_add >>> (A+shiftX, A+shiftX+shiftY, C+shiftX, widthA, widthA, widthC);
		add <<< grid_add, thread_add >>> (B+shiftY, B+shiftX+shiftY, C+shiftY, widthB, widthB, widthC);
		
		strassen_paper_recursive(C+shiftX, C+shiftY, C, subWidth, widthC, widthC, widthC);

		add <<< grid_add, thread_add >>> (A, A+shiftX+shiftY, C+shiftX, widthA, widthA, widthC);
		add <<< grid_add, thread_add >>> (B, B+shiftX+shiftY, C+shiftY, widthB, widthB, widthC);
		
		strassen_paper_recursive(C+shiftX, C+shiftY, t1, subWidth, widthC, widthC, subWidth);

		add_add <<< grid_add, thread_add >>> (t1, C, C+shiftX+shiftY, subWidth, widthC, widthC);
		add <<< grid_add, thread_add >>> (A+shiftY, A+shiftX+shiftY, t2, widthA, widthA, subWidth);

		strassen_paper_recursive(t2, B, C+shiftY, subWidth, subWidth, widthB, widthC);

		sub <<< grid_add, thread_add >>> (C+shiftX+shiftY, C+shiftY, C+shiftX+shiftY, widthC, widthC, widthC);
		sub <<< grid_add, thread_add >>> (B+shiftY, B, t1, widthB, widthB, subWidth);

		strassen_paper_recursive(A+shiftX+shiftY, t1, t2, subWidth, widthA, subWidth, subWidth);
	
		add_add <<< grid_add, thread_add >>> (t2, C, C+shiftY, subWidth, widthC, widthC);
		sub <<< grid_add, thread_add >>> (B+shiftX, B+shiftX+shiftY, t1, widthB, widthB, subWidth);

		strassen_paper_recursive(A, t1, C+shiftX, subWidth, widthA, subWidth, widthC);

		add <<< grid_add, thread_add >>> (C+shiftX+shiftY, C+shiftX, C+shiftX+shiftY, widthC, widthC, widthC);
		add <<< grid_add, thread_add >>> (A, A+shiftX, t2, widthA, widthA, subWidth);

		strassen_paper_recursive(t2, B+shiftX+shiftY, t1, subWidth, subWidth, widthB, subWidth);

		add_sub <<< grid_add, thread_add >>> (t1, C+shiftX, C, subWidth, widthC, widthC);

		cudaFree(t1);
                cudaFree(t2);
	}

}

void strassen (float *A, float *B, float *C, int width)
{

	float *m1, *m2, *m3, *m4, *m5, *m6, *m7;
	float *m1a, *m1b, *m2a, *m2b, *m3a, *m3b, *m4a, *m4b, *m5a, *m5b, *m6a, *m6b, *m7a, *m7b;

	int subWidth = width/2;
	int memSize = subWidth*subWidth*sizeof(float);

	int shiftX = subWidth;
	int shiftY = width*subWidth;

    cudaMalloc((void**) &m1, memSize);
    cudaMalloc((void**) &m2, memSize);
    cudaMalloc((void**) &m3, memSize);
    cudaMalloc((void**) &m4, memSize);
    cudaMalloc((void**) &m5, memSize);
    cudaMalloc((void**) &m6, memSize);
    cudaMalloc((void**) &m7, memSize);

    checkCudaErrors(cudaMalloc((void**) &m1a, memSize));
    checkCudaErrors(cudaMalloc((void**) &m1b, memSize));
    checkCudaErrors(cudaMalloc((void**) &m2a, memSize));
    checkCudaErrors(cudaMalloc((void**) &m3b, memSize));
    checkCudaErrors(cudaMalloc((void**) &m4b, memSize));
    checkCudaErrors(cudaMalloc((void**) &m5a, memSize));
    checkCudaErrors(cudaMalloc((void**) &m6a, memSize));
    checkCudaErrors(cudaMalloc((void**) &m6b, memSize));
    checkCudaErrors(cudaMalloc((void**) &m7a, memSize));
    checkCudaErrors(cudaMalloc((void**) &m7b, memSize));
#ifdef MULTILEVEL
    checkCudaErrors(cudaMalloc((void**) &m2b, memSize));
    checkCudaErrors(cudaMalloc((void**) &m3a, memSize));
    checkCudaErrors(cudaMalloc((void**) &m4a, memSize));
    checkCudaErrors(cudaMalloc((void**) &m5b, memSize));
#endif

	dim3 threads(BLOCK_X, BLOCK_Y);
//	dim3 grids(subWidth/threads.x, subWidth/threads.y);
	dim3 grid(subWidth/TILE_X, subWidth/TILE_Y);
	dim3 threads_add(TILE_X,TILE_Y), grid_add(subWidth/threads_add.x, subWidth/threads_add.y);
	dim3 threads_mul(256, 1);
	dim3 grid_mul(subWidth/threads_mul.x, subWidth/threads_mul.y/16);

//    printf("Recursion: subWidth = %d, block dim = (%d, %d), grid dim = (%d, %d)\n", subWidth, threads.x, threads.y, grid.x, grid.y);

    cutilCheckMsg("before kernels");
	addition <<< grid_add,threads_add >>> (m1a, A, A+shiftX+shiftY, width, subWidth, subWidth);
    cutilCheckMsg("before second kernel");
	addition <<< grid_add,threads_add >>> (m1b, B, B+shiftX+shiftY, width, subWidth, subWidth);
	addition <<< grid_add,threads_add >>> (m2a, A+shiftY, A+shiftX+shiftY, width, subWidth, subWidth);

#ifdef MULTILEVEL
	subMatrix <<< grid,threads >>> (m2b, B, subWidth, width);
	subMatrix <<< grid,threads >>> (m3a, A, subWidth, width);
#endif
	subtraction <<< grid_add,threads_add >>> (m3b, B+shiftX, B+shiftX+shiftY, width, width, subWidth, subWidth);
#ifdef MULTILEVEL
	subMatrix <<< grid,threads >>> (m4a, A+shiftX+shiftY, subWidth, width);
#endif
	subtraction <<< grid_add,threads_add >>> (m4b, B+shiftY, B, width, width, subWidth, subWidth);
	addition <<< grid_add,threads_add >>> (m5a, A, A+shiftX, width, subWidth, subWidth);
    cutilCheckMsg("before sub matrix");
#ifdef MULTILEVEL
	subMatrix <<< grid,threads >>> (m5b, B+shiftX+shiftY, subWidth, width);
#endif
    cutilCheckMsg("before sub add");
	subtraction <<< grid_add,threads_add >>> (m6a, A+shiftY, A, width, width, subWidth, subWidth);
    cutilCheckMsg("after first sub");
	addition <<< grid_add,threads_add >>> (m6b, B, B+shiftX, width, subWidth, subWidth);
    cutilCheckMsg("after first add");
	subtraction <<< grid_add,threads_add >>> (m7a, A+shiftX, A+shiftX+shiftY, width, width, subWidth, subWidth);
    cutilCheckMsg("after second sub");
	addition <<< grid_add,threads_add >>> (m7b, B+shiftY, B+shiftX+shiftY, width, subWidth, subWidth);
    cutilCheckMsg("after second add");
	checkCudaErrors(cudaThreadSynchronize());
    cutilCheckMsg("going to matmul");
	
#ifdef MULTILEVEL
	if(width <= BLOCK_THRESHOLD){
/*
	        multiplication <<< grid,threads >>> (m1, m1a, m1b, subWidth, subWidth, subWidth);
        	multiplication <<< grid,threads >>> (m2, m2a, B, subWidth, width, subWidth);
	        multiplication <<< grid,threads >>> (m3, A, m3b, width, subWidth, subWidth);
        	multiplication <<< grid,threads >>> (m4, A+shiftX+shiftY, m4b, width, subWidth, subWidth);
	        multiplication <<< grid,threads >>> (m5, m5a, B+shiftX+shiftY, subWidth, width, subWidth);
        	multiplication <<< grid,threads >>> (m6, m6a, m6b, subWidth, subWidth, subWidth);
	        multiplication <<< grid,threads >>> (m7, m7a, m7b, subWidth, subWidth, subWidth);
*/
    cutilCheckMsg("before matmul");
		matmul_opt <<< grid_mul, threads_mul >>> (m1a, m1b, m1, subWidth, subWidth, subWidth, subWidth);
    cutilCheckMsg("before second matmul");
		matmul_opt <<< grid_mul, threads_mul >>> (m2a, B, m2, subWidth, subWidth, width, subWidth);
		matmul_opt <<< grid_mul, threads_mul >>> (A, m3b, m3, subWidth, width, subWidth, subWidth);
		matmul_opt <<< grid_mul, threads_mul >>> (A+shiftX+shiftY, m4b, m4, subWidth, width, subWidth, subWidth);
		matmul_opt <<< grid_mul, threads_mul >>> (m5a, B+shiftX+shiftY, m5, subWidth, subWidth, width, subWidth);
		matmul_opt <<< grid_mul, threads_mul >>> (m6a, m6b, m6, subWidth, subWidth, subWidth, subWidth);
		matmul_opt <<< grid_mul, threads_mul >>> (m7a, m7b, m7, subWidth, subWidth, subWidth, subWidth);
	}
	else{
		strassen(m1a, m1b, m1, subWidth);
		strassen(m2a, m2b, m2, subWidth); 
		strassen(m3a, m3b, m3, subWidth);
		strassen(m4a, m4b, m4, subWidth);
		strassen(m5a, m5b, m5, subWidth);
		strassen(m6a, m6b, m6, subWidth);
		strassen(m7a, m7b, m7, subWidth);
	}
#else
/*
	multiplication <<< grid,threads >>> (m1, m1a, m1b, subWidth, subWidth, subWidth);
	multiplication <<< grid,threads >>> (m2, m2a, B, subWidth, width, subWidth);
	multiplication <<< grid,threads >>> (m3, A, m3b, width, subWidth, subWidth);
	multiplication <<< grid,threads >>> (m4, A+shiftX+shiftY, m4b, width, subWidth, subWidth);
	multiplication <<< grid,threads >>> (m5, m5a, B+shiftX+shiftY, subWidth, width, subWidth);
	multiplication <<< grid,threads >>> (m6, m6a, m6b, subWidth, subWidth, subWidth);
	multiplication <<< grid,threads >>> (m7, m7a, m7b, subWidth, subWidth, subWidth);
*/
		matmul_opt <<< grid_mul, threads_mul >>> (m1a, m1b, m1, subWidth, subWidth, subWidth, subWidth);
		matmul_opt <<< grid_mul, threads_mul >>> (m2a, B, m2, subWidth, subWidth, width, subWidth);
		matmul_opt <<< grid_mul, threads_mul >>> (A, m3b, m3, subWidth, width, subWidth, subWidth);
		matmul_opt <<< grid_mul, threads_mul >>> (A+shiftX+shiftY, m4b, m4, subWidth, width, subWidth, subWidth);
		matmul_opt <<< grid_mul, threads_mul >>> (m5a, B+shiftX+shiftY, m5, subWidth, subWidth, width, subWidth);
		matmul_opt <<< grid_mul, threads_mul >>> (m6a, m6b, m6, subWidth, subWidth, subWidth, subWidth);
		matmul_opt <<< grid_mul, threads_mul >>> (m7a, m7b, m7, subWidth, subWidth, subWidth, subWidth);

#endif

    cutilCheckMsg("before compute");
	checkCudaErrors(cudaThreadSynchronize());
#ifdef STRASSEN_WITH_GPGPUCOMPILER_KERNELS
	dim3 threads_compute(16, 1);
	dim3 grid_compute(subWidth/threads_compute.x, subWidth/threads_compute.y/16);
	computeC11 <<< grid_compute,threads_compute >>> (C, m1, m4, m5, m7, width, subWidth);
	computeC12 <<< grid_compute,threads_compute >>> (C+shiftX, m3, m5, width, subWidth);
	computeC21 <<< grid_compute,threads_compute >>> (C+shiftY, m2, m4, width, subWidth);
	computeC22 <<< grid_compute,threads_compute >>> (C+shiftX+shiftY, m1, m2, m3, m6, width,subWidth);
#else
	computeC11 <<< grid,threads >>> (C, m1, m4, m5, m7, width, subWidth);
	computeC12 <<< grid,threads >>> (C+shiftX, m3, m5, width, subWidth);
	computeC21 <<< grid,threads >>> (C+shiftY, m2, m4, width, subWidth);
	computeC22 <<< grid,threads >>> (C+shiftX+shiftY, m1, m2, m3, m6, width,subWidth);
#endif
	checkCudaErrors(cudaThreadSynchronize());

    checkCudaErrors(cudaFree(m1));
    checkCudaErrors(cudaFree(m2));
    checkCudaErrors(cudaFree(m3));
    checkCudaErrors(cudaFree(m4));
    checkCudaErrors(cudaFree(m5));
    checkCudaErrors(cudaFree(m6));
    checkCudaErrors(cudaFree(m7));
    checkCudaErrors(cudaFree(m1a));
    checkCudaErrors(cudaFree(m1b));
    checkCudaErrors(cudaFree(m2a));
    checkCudaErrors(cudaFree(m3b));
    checkCudaErrors(cudaFree(m4b));
    checkCudaErrors(cudaFree(m5a));
    checkCudaErrors(cudaFree(m6a));
    checkCudaErrors(cudaFree(m6b));
    checkCudaErrors(cudaFree(m7a));
    checkCudaErrors(cudaFree(m7b));
#ifdef MULTILEVEL
    checkCudaErrors(cudaFree(m2b));
    checkCudaErrors(cudaFree(m3a));
    checkCudaErrors(cudaFree(m4a));
    checkCudaErrors(cudaFree(m5b));
#endif
}

void strassen_with_cublas (float *A, float *B, float *C, int width, cublasHandle_t *handle, const float *alpha, const float *beta)
{

	float *m1, *m2, *m3, *m4, *m5, *m6, *m7;
	float *m1a, *m1b, *m2a, *m2b, *m3a, *m3b, *m4a, *m4b, *m5a, *m5b, *m6a, *m6b, *m7a, *m7b;

	int subWidth = width/2;
	int memSize = subWidth*subWidth*sizeof(float);

	int shiftX = subWidth;
	int shiftY = width*subWidth;

    checkCudaErrors(cudaMalloc((void**) &m1, memSize));
    checkCudaErrors(cudaMalloc((void**) &m2, memSize));
    checkCudaErrors(cudaMalloc((void**) &m3, memSize));
    checkCudaErrors(cudaMalloc((void**) &m4, memSize));
    checkCudaErrors(cudaMalloc((void**) &m5, memSize));
    checkCudaErrors(cudaMalloc((void**) &m6, memSize));
    checkCudaErrors(cudaMalloc((void**) &m7, memSize));

    checkCudaErrors(cudaMalloc((void**) &m1a, memSize));
    checkCudaErrors(cudaMalloc((void**) &m1b, memSize));
    checkCudaErrors(cudaMalloc((void**) &m2a, memSize));
    checkCudaErrors(cudaMalloc((void**) &m3b, memSize));
    checkCudaErrors(cudaMalloc((void**) &m4b, memSize));
    checkCudaErrors(cudaMalloc((void**) &m5a, memSize));
    checkCudaErrors(cudaMalloc((void**) &m6a, memSize));
    checkCudaErrors(cudaMalloc((void**) &m6b, memSize));
    checkCudaErrors(cudaMalloc((void**) &m7a, memSize));
    checkCudaErrors(cudaMalloc((void**) &m7b, memSize));
#ifdef MULTILEVEL
    checkCudaErrors(cudaMalloc((void**) &m2b, memSize));
    checkCudaErrors(cudaMalloc((void**) &m3a, memSize));
    checkCudaErrors(cudaMalloc((void**) &m4a, memSize));
    checkCudaErrors(cudaMalloc((void**) &m5b, memSize));
#endif

	dim3 threads(BLOCK_X, BLOCK_Y);
//	dim3 grids(subWidth/threads.x, subWidth/threads.y);
	dim3 grid(subWidth/TILE_X, subWidth/TILE_Y);
	dim3 threads_add(TILE_X,TILE_Y), grid_add(subWidth/threads_add.x, subWidth/threads_add.y);
	dim3 threads_mul(256, 1);
	dim3 grid_mul(subWidth/threads_mul.x, subWidth/threads_mul.y/16);

#ifndef STRASSEN_WITH_GPGPUCOMPILER_KERNELS
	threads_add = threads; grid_add = grid;
#endif
	
//	printf("Recursion: block dim = (%d, %d), grid dim = (%d, %d)\n", threads.x, threads.y, grids.x, grids.y);

	cublasSgeam(*handle, CUBLAS_OP_N, CUBLAS_OP_N, subWidth, subWidth, alpha, A, width, alpha, A+shiftX+shiftY, width, m1a, subWidth);
	cublasSgeam(*handle, CUBLAS_OP_N, CUBLAS_OP_N, subWidth, subWidth, alpha, B, width, alpha, B+shiftX+shiftY, width, m1b, subWidth);
	cublasSgeam(*handle, CUBLAS_OP_N, CUBLAS_OP_N, subWidth, subWidth, alpha, A+shiftY, width, alpha, A+shiftX+shiftY, width, m2a, subWidth);
/*
	addition <<< grid_add,threads_add >>> (m1a, A, A+shiftX+shiftY, width, subWidth, subWidth);
	addition <<< grid_add,threads_add >>> (m1b, B, B+shiftX+shiftY, width, subWidth, subWidth);
	addition <<< grid_add,threads_add >>> (m2a, A+shiftY, A+shiftX+shiftY, width, subWidth, subWidth);
*/
#ifdef MULTILEVEL
	cublasSgeam(*handle, CUBLAS_OP_N, CUBLAS_OP_N, subWidth, subWidth, alpha, B, width, beta, B, width, m2b, subWidth);
	cublasSgeam(*handle, CUBLAS_OP_N, CUBLAS_OP_N, subWidth, subWidth, alpha, A, width, beta, A, width, m3a, subWidth);
//	subMatrix <<< grid,threads >>> (m2b, B, subWidth, width);
//	subMatrix <<< grid,threads >>> (m3a, A, subWidth, width);
#endif
	subtraction <<< grid_add,threads_add >>> (m3b, B+shiftX, B+shiftX+shiftY, width, width, subWidth, subWidth);
#ifdef MULTILEVEL
	cublasSgeam(*handle, CUBLAS_OP_N, CUBLAS_OP_N, subWidth, subWidth, alpha, A+shiftX+shiftY, width, beta, A+shiftX+shiftY, width, m4a, subWidth);
//	subMatrix <<< grid,threads >>> (m4a, A+shiftX+shiftY, subWidth, width);
#endif
	subtraction <<< grid_add,threads_add >>> (m4b, B+shiftY, B, width, width, subWidth, subWidth);
	cublasSgeam(*handle, CUBLAS_OP_N, CUBLAS_OP_N, subWidth, subWidth, alpha, A, width, alpha, A+shiftX, width, m5a, subWidth);
//	addition <<< grid_add,threads_add >>> (m5a, A, A+shiftX, width, subWidth, subWidth);
#ifdef MULTILEVEL
	cublasSgeam(*handle, CUBLAS_OP_N, CUBLAS_OP_N, subWidth, subWidth, alpha, B+shiftX+shiftY, width, beta, B+shiftX+shiftY, width, m5b, subWidth);
//	subMatrix <<< grid,threads >>> (m5b, B+shiftX+shiftY, subWidth, width);
#endif
	subtraction <<< grid_add,threads_add >>> (m6a, A+shiftY, A, width, width, subWidth, subWidth);
	cublasSgeam(*handle, CUBLAS_OP_N, CUBLAS_OP_N, subWidth, subWidth, alpha, B, width, alpha, B+shiftX, width, m6b, subWidth);
//	addition <<< grid_add,threads_add >>> (m6b, B, B+shiftX, width, subWidth, subWidth);
	subtraction <<< grid_add,threads_add >>> (m7a, A+shiftX, A+shiftX+shiftY, width, width, subWidth, subWidth);
	cublasSgeam(*handle, CUBLAS_OP_N, CUBLAS_OP_N, subWidth, subWidth, alpha, B+shiftY, width, alpha, B+shiftX+shiftY, width, m7b, subWidth);
//	addition <<< grid_add,threads_add >>> (m7b, B+shiftY, B+shiftX+shiftY, width, subWidth, subWidth);
	cudaThreadSynchronize();
	
#ifdef MULTILEVEL
	if(width <= BLOCK_THRESHOLD){

	       	cublasSgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_N, subWidth, subWidth, subWidth, alpha, m1b, subWidth, m1a, subWidth, beta, m1, subWidth);

        	cublasSgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_N, subWidth, subWidth, subWidth, alpha, B, width, m2a, subWidth, beta, m2, subWidth);
        	cublasSgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_N, subWidth, subWidth, subWidth, alpha, m3b, subWidth, A, width, beta, m3, subWidth);
        	cublasSgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_N, subWidth, subWidth, subWidth, alpha, m4b, subWidth, A+shiftX+shiftY, width, beta, m4, subWidth);
        	cublasSgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_N, subWidth, subWidth, subWidth, alpha, B+shiftX+shiftY, width, m5a, subWidth, beta, m5, subWidth);
        	cublasSgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_N, subWidth, subWidth, subWidth, alpha, m6b, subWidth, m6a, subWidth, beta, m6, subWidth);
        	cublasSgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_N, subWidth, subWidth, subWidth, alpha, m7b, subWidth, m7a, subWidth, beta, m7, subWidth);
				
/*
		matmul_opt <<< grid_mul, threads_mul >>> (m1a, m1b, m1, subWidth, subWidth, subWidth, subWidth);
		matmul_opt <<< grid_mul, threads_mul >>> (m2a, B, m2, subWidth, subWidth, width, subWidth);
		matmul_opt <<< grid_mul, threads_mul >>> (A, m3b, m3, subWidth, width, subWidth, subWidth);
		matmul_opt <<< grid_mul, threads_mul >>> (A+shiftX+shiftY, m4b, m4, subWidth, width, subWidth, subWidth);
		matmul_opt <<< grid_mul, threads_mul >>> (m5a, B+shiftX+shiftY, m5, subWidth, subWidth, width, subWidth);
		matmul_opt <<< grid_mul, threads_mul >>> (m6a, m6b, m6, subWidth, subWidth, subWidth, subWidth);
		matmul_opt <<< grid_mul, threads_mul >>> (m7a, m7b, m7, subWidth, subWidth, subWidth, subWidth);
*/
	}
	else{
		strassen(m1a, m1b, m1, subWidth);
		strassen(m2a, m2b, m2, subWidth); 
		strassen(m3a, m3b, m3, subWidth);
		strassen(m4a, m4b, m4, subWidth);
		strassen(m5a, m5b, m5, subWidth);
		strassen(m6a, m6b, m6, subWidth);
		strassen(m7a, m7b, m7, subWidth);
	}
#else
/*
	multiplication <<< grid,threads >>> (m1, m1a, m1b, subWidth, subWidth, subWidth);
	multiplication <<< grid,threads >>> (m2, m2a, B, subWidth, width, subWidth);
	multiplication <<< grid,threads >>> (m3, A, m3b, width, subWidth, subWidth);
	multiplication <<< grid,threads >>> (m4, A+shiftX+shiftY, m4b, width, subWidth, subWidth);
	multiplication <<< grid,threads >>> (m5, m5a, B+shiftX+shiftY, subWidth, width, subWidth);
	multiplication <<< grid,threads >>> (m6, m6a, m6b, subWidth, subWidth, subWidth);
	multiplication <<< grid,threads >>> (m7, m7a, m7b, subWidth, subWidth, subWidth);
*/
		matmul_opt <<< grid_mul, threads_mul >>> (m1a, m1b, m1, subWidth, subWidth, subWidth, subWidth);
		matmul_opt <<< grid_mul, threads_mul >>> (m2a, B, m2, subWidth, subWidth, width, subWidth);
		matmul_opt <<< grid_mul, threads_mul >>> (A, m3b, m3, subWidth, width, subWidth, subWidth);
		matmul_opt <<< grid_mul, threads_mul >>> (A+shiftX+shiftY, m4b, m4, subWidth, width, subWidth, subWidth);
		matmul_opt <<< grid_mul, threads_mul >>> (m5a, B+shiftX+shiftY, m5, subWidth, subWidth, width, subWidth);
		matmul_opt <<< grid_mul, threads_mul >>> (m6a, m6b, m6, subWidth, subWidth, subWidth, subWidth);
		matmul_opt <<< grid_mul, threads_mul >>> (m7a, m7b, m7, subWidth, subWidth, subWidth, subWidth);

#endif

	cudaThreadSynchronize();
//	cublasSgeam(*handle, CUBLAS_OP_N, CUBLAS_OP_N, subWidth, subWidth, alpha, m1, subWidth, alpha, m4, subWidth, C, width);
//	subtraction <<< grid_add,threads_add >>> (C, C, m5, width, subWidth, width, subWidth);
//	cublasSgeam(*handle, CUBLAS_OP_N, CUBLAS_OP_N, subWidth, subWidth, alpha, C, width, alpha, m7, subWidth, C, width);
	computeC11 <<< grid,threads >>> (C, m1, m4, m5, m7, width, subWidth);
	computeC12 <<< grid,threads >>> (C+shiftX, m3, m5, width, subWidth);
	computeC21 <<< grid,threads >>> (C+shiftY, m2, m4, width, subWidth);
	computeC22 <<< grid,threads >>> (C+shiftX+shiftY, m1, m2, m3, m6, width,subWidth);
	cudaThreadSynchronize();

    checkCudaErrors(cudaFree(m1));
    checkCudaErrors(cudaFree(m2));
    checkCudaErrors(cudaFree(m3));
    checkCudaErrors(cudaFree(m4));
    checkCudaErrors(cudaFree(m5));
    checkCudaErrors(cudaFree(m6));
    checkCudaErrors(cudaFree(m7));
    checkCudaErrors(cudaFree(m1a));
    checkCudaErrors(cudaFree(m1b));
    checkCudaErrors(cudaFree(m2a));
    checkCudaErrors(cudaFree(m3b));
    checkCudaErrors(cudaFree(m4b));
    checkCudaErrors(cudaFree(m5a));
    checkCudaErrors(cudaFree(m6a));
    checkCudaErrors(cudaFree(m6b));
    checkCudaErrors(cudaFree(m7a));
    checkCudaErrors(cudaFree(m7b));
#ifdef MULTILEVEL
    checkCudaErrors(cudaFree(m2b));
    checkCudaErrors(cudaFree(m3a));
    checkCudaErrors(cudaFree(m4a));
    checkCudaErrors(cudaFree(m5b));
#endif
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char** argv)
{

    int devID = findCudaDevice(argc, (const char **)argv);
    cudaDeviceProp props;

    // get number of SMs on this GPU
    checkCudaErrors(cudaGetDevice(&devID));
    checkCudaErrors(cudaGetDeviceProperties(&props, devID));

    printf("Device %d: \"%s\" with Compute %d.%d capability, SM's = %d\n", devID, props.name, props.major, props.minor, props.multiProcessorCount);

	// set seed for rand()
    srand(2006);

    // Optional Command-line multiplier for matrix sizes
    unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
    int iSizeMultiple = 1;
		uiWA = WA * iSizeMultiple;
		uiHA = HA * iSizeMultiple;
		uiWB = WB * iSizeMultiple;
		uiHB = HB * iSizeMultiple;
		uiWC = WC * iSizeMultiple;
		uiHC = HC * iSizeMultiple;
    shrLog("\nUsing Matrix Sizes: A(%u x %u), B(%u x %u), C(%u x %u)\n\n", 
            uiWA, uiHA, uiWB, uiHB, uiWC, uiHC);

    // allocate host memory for matrices A and B
    unsigned int size_A = uiWA * uiHA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A = (float*)malloc(mem_size_A);
    unsigned int size_B = uiWB * uiHB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B = (float*)malloc(mem_size_B);

    // initialize host memory
    randomInit(h_A, size_A);
    randomInit(h_B, size_B);

    // allocate device memory
    float* d_A;
    checkCudaErrors(cudaMalloc((void**) &d_A, mem_size_A));
    float* d_B;
    checkCudaErrors(cudaMalloc((void**) &d_B, mem_size_B));

    // copy host memory to device
    checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A,
                              cudaMemcpyHostToDevice) );
    checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B,
                              cudaMemcpyHostToDevice) );

    // allocate device memory for result
    unsigned int size_C = uiWC * uiHC;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float* d_C;
    checkCudaErrors(cudaMalloc((void**) &d_C, mem_size_C));

    // allocate host memory for the result
    float* h_C = (float*) malloc(mem_size_C);

    int nIter = 10;
    double dSeconds;
//    double dNumOps;
//    double gflops;
    unsigned int timer = 0;

    float* reference = (float*)malloc(mem_size_C);
/*
    // compute reference solution
    shrLog("\nHost computation...\n\n");    

    cutilCheckError(cutCreateTimer(&timer));
    cutilCheckError(cutStartTimer(timer));

    computeGold(reference, h_A, h_B, uiHA, uiWA, uiWB);

    cutilCheckError(cutStopTimer(timer));
    dSeconds = cutGetTimerValue(timer)/((double)nIter * 1000.0);
    dNumOps = 2.0 * (double)uiWA * (double)uiHA * (double)uiWB;
    gflops = 1.0e-9 * dNumOps/dSeconds;

    //Log througput, etc
    shrLogEx("matrixMul Host Only, Throughput = %.4f GFlop/s, Time = %.5f s, Size = %.0f Ops, NumDevsUsed = %d, Workgroup = %u\n", 
            gflops, dSeconds, dNumOps, 1, 1);
    cutilCheckError(cutDeleteTimer(timer));
*/
/*
// GPGPU Compiler Generated Kernel
    // setup execution parameters
        dim3 threadsG(256, 1);
    dim3 gridG(ASIZE/threadsG.x, ASIZE/threadsG.y/16);

        timer = 0;
	cutilCheckError(cutCreateTimer(&timer));
	cutilCheckError(cutStartTimer(timer));
  //====================================================================
    // execute the kernel
    for (int j = 0; j < nIter; j++)
                {
            matmul_opt<<< gridG, threadsG >>>(d_A, d_B, d_C, ASIZE, ASIZE, ASIZE, ASIZE);
                    cudaThreadSynchronize();
        }
  //====================================================================

  CUT_CHECK_ERROR("Kernel execution failed");

  cutilCheckError(cutStopTimer(timer));
  dSeconds = cutGetTimerValue(timer)/((double)nIter * 1000.0);

  printf("GPGPU Compiler Elapsed Time: %f s\n", dSeconds);
  cutilCheckError(cutDeleteTimer(timer));

   cudaMemcpy(reference, d_C, mem_size_C, cudaMemcpyDeviceToHost);

//End of GPGPU Compiler Generated Kernel
*/
/*
	if(check_result(reference, h_C, size_C) == 0)
        	printf("PASSED\n");
	else
        	printf("FAILED\n");

*/

// CUBLAS version 3.0
        cublasHandle_t handle;
        checkError(cublasCreate(&handle), "cublasCreate() error!\n");
	int version;
	cublasGetVersion(handle, &version);
	printf("Cublas Library Version = %d\n", version); 
        const float alpha = 1.0f;
        const float beta = 0.0f;
        //Perform warmup operation with cublas
        cublasStatus_t ret = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, uiWB, uiHA, uiWA, &alpha, d_B, uiWB, d_A, uiWA, &beta, d_C, uiWA);
        checkError(ret, "cublas Sgemm returned an error!\n");

		// Start Timing
        timer = 0;
	cutilCheckError(cutCreateTimer(&timer));
	cutilCheckError(cutStartTimer(timer));
        for (int j = 0; j < nIter; j++) {
            //note cublas is column primary!
            //need to transpose the order
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, uiWB, uiHA, uiWA, &alpha, d_B, uiWB, d_A, uiWA, &beta, d_C, uiWA);
		}
		// check if kernel execution generated and error
		getLastCudaError("CUBLAS Kernel execution failed");
		cudaDeviceSynchronize();
		// stop and destroy timer
		  cutilCheckError(cutStopTimer(timer));
		  dSeconds = cutGetTimerValue(timer)/((double)nIter * 1000.0);
//		double dNumOps = 2.0 * (double)uiWA * (double)uiHA * (double)uiWB;
//		double gflops = 1.0e-9 * dNumOps/dSeconds;

		//Log througput, etc
		    shrLogEx("cublas, Time = %.5f s\n", dSeconds);
		    cutilCheckError(cutDeleteTimer(timer));

		// copy result from device to host
		checkCudaErrors(cudaMemcpy(reference, d_C, mem_size_C, cudaMemcpyDeviceToHost) );

//        checkError(cublasDestroy(handle), "cublasDestroy() error!\n");
//End of Cublas v 3.0
/*
	if(check_result(reference, h_C, size_C) == 0)
        	printf("PASSED\n");
	else
        	printf("FAILED\n");
*/
/*
//Strassen Kernel

    // create and start timer
    shrLog("Run Strassen Kernels...\n\n");
	timer = 0;
    cutilCheckError(cutCreateTimer(&timer));
    cutilCheckError(cutStartTimer(timer));

    // execute the kernel
    for (int j = 0; j < nIter; j++) 
		{
			strassen(d_A, d_B, d_C, N);
        }

    // check if kernel execution generated and error
    cutilCheckMsg("Kernel execution failed");

    // stop and destroy timer
    cutilCheckError(cutStopTimer(timer));
    dSeconds = cutGetTimerValue(timer)/((double)nIter * 1000.0);
//    dNumOps = 2.0 * (double)uiWA * (double)uiHA * (double)uiWB;
//    gflops = 1.0e-9 * dNumOps/dSeconds;

    //Log througput, etc
    shrLogEx("matrixMul_strassen, Time = %.5f s\n", dSeconds);
    cutilCheckError(cutDeleteTimer(timer));
    // copy result from device to host
    checkCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost) );

//End of Strassen

	if(check_result(reference, h_C, size_C) == 0)
        	printf("PASSED\n");
	else
        	printf("FAILED\n");
*/
/*
//Strassen Kernel paper version

    // create and start timer
    shrLog("Run Strassen Kernels paper version...\n\n");
        timer = 0;
    cutilCheckError(cutCreateTimer(&timer));
    cutilCheckError(cutStartTimer(timer));

    // execute the kernel
    for (int j = 0; j < nIter; j++)
                {
                        strassen_paper_recursive(d_A, d_B, d_C, N, N, N, N);
			cudaThreadSynchronize();
        }

//    cutilCheckMsg("Kernel execution failed");

    // stop and destroy timer
    cutilCheckError(cutStopTimer(timer));
    dSeconds = cutGetTimerValue(timer)/((double)nIter * 1000.0);
//    dNumOps = 2.0 * (double)uiWA * (double)uiHA * (double)uiWB;
//    gflops = 1.0e-9 * dNumOps/dSeconds;

    //Log througput, etc
    shrLogEx("matrixMul_strassen paper version, Time = %.5f s\n", dSeconds);
    cutilCheckError(cutDeleteTimer(timer));
    // copy result from device to host
    cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

//End of Strassen Paper Version

    // check result

	if(check_result(reference, h_C, size_C) == 0)
        	printf("PASSED\n");
	else
        	printf("FAILED\n");
*/

//Strassen Kernel with CUBLAS

    // create and start timer
    shrLog("Run Strassen Kernels using CUBLAS...\n\n");
//       	cublasHandle_t shandle;
//        checkError(cublasCreate(&shandle), "cublasCreate() error!\n");
        const float salpha = 1.0f;
        const float sbeta = 0.0f;
	timer = 0;
    cutilCheckError(cutCreateTimer(&timer));
    cutilCheckError(cutStartTimer(timer));

    // execute the kernel
    for (int j = 0; j < nIter; j++) 
		{
			strassen_with_cublas(d_A, d_B, d_C, N, &handle, &salpha, &sbeta);
        }

    // check if kernel execution generated and error
    cutilCheckMsg("Kernel execution failed");

    // stop and destroy timer
    cutilCheckError(cutStopTimer(timer));
    dSeconds = cutGetTimerValue(timer)/((double)nIter * 1000.0);
//    dNumOps = 2.0 * (double)uiWA * (double)uiHA * (double)uiWB;
//    gflops = 1.0e-9 * dNumOps/dSeconds;

    //Log througput, etc
    shrLogEx("matrixMul_strassen_with_cublas, Time = %.5f s\n", dSeconds);
    cutilCheckError(cutDeleteTimer(timer));
    // copy result from device to host
    checkCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost) );
//        checkError(cublasDestroy(shandle), "cublasDestroy() error!\n");
        checkError(cublasDestroy(handle), "cublasDestroy() error!\n");

//End of Strassen with CUBLAS

	if(check_result(reference, h_C, size_C) == 0)
        	printf("PASSED\n");
	else
        	printf("FAILED\n");

	shrLog("End of Program\n");


    // clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(reference);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaThreadExit();
}

// Allocates a matrix with random float entries.
void randomInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

void printDiff(float *data1, float *data2, int width, int height, int 
iListLength, float fListTol)
{
    shrLog("Listing first %d Differences > %.6f...\n", iListLength, 
fListTol);
    int i,j,k;
    int error_count=0;
    for (j = 0; j < height; j++) 
    {
        if (error_count < iListLength)
        {
            shrLog("\n  Row %d:\n", j);
        }
        for (i = 0; i < width; i++) 
        {
            k = j * width + i;
            float fDiff = fabs(data1[k] - data2[k]);
            if (fDiff > fListTol) 
            {                
                if (error_count < iListLength)
                {
                    shrLog("Loc(%d,%d)\tCPU=%.5f\tGPU=%.5f\tDiff=%.6f\n", i, j, data1[k], data2[k], fDiff);
                }
                error_count++;
            }
        }
    }
    shrLog(" \n  Total Errors = %d\n\n", error_count);
}

