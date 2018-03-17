#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

//matrix dimensions
/* DIM_N defines matix size, threads sets the openmp thread count      * 
 * threshold sets the lower limit for the strassen recursion algorithm *
 * chunk defines the openmp chunking size                              */
int DIM_N =1024;
int threads= 32;
int threshold= 1024;

//other stuff
double sum, snorm;



//matrices
double **A, **B, **C,**CC;

//Prototypes
void strassenMultMatrix(double**,double**,double**,int,int,int,int,int,int,int);
void normalMultMatrix(double**, double**, double**, int,int,int,int,int,int,int);
void subMatrices(double**, double**, double**, int,int,int,int,int);
void subMatrices1(double**, double**, double**, int,int,int,int,int);
void subMatricesc(double**, double**, double**, int,int,int,int,int,int,int);
void addMatrices(double**, double**, double**, int,int,int,int,int);
void addMatrices1(double**, double**, double**, int,int,int,int,int);
void addMatricesc(double**, double**, double**, int,int,int,int,int,int,int);
void myprint(double **,char *,int,int ,int);
//##########################################################
//Error calculation
//Following functions can be used to check numerical errors:
 
void checkPracticalErrors (double **c, double **seq, int n)
{
int ii;
int n2 = n*n;
double sum =0;
double low = c[0][0] - seq[0][0];
double up = low;
for (ii=0; ii<n2; ii++)
{
int i = ii / n;
int j = ii % n;
double temp = c[i][j] - seq[i][j];
sum += (temp<0 ? -temp: temp);
if (temp > up)
up = temp;
else if (temp< low)
low = temp;
}
printf ("average error: %.20f\n", sum/n2);
printf ("lower-bound: %.20f\n", low);
printf ("upper-bound: %.20f\n", up);
printf ("\n");
}

void accuracyTestInit (double** a, double **b, int n)
{
int i,j;
double *uvT = (double *) malloc ( n*n*sizeof(double) );
//initiate a and b
for (i =0 ; i< n; i++)
{
for (j =0; j< n; j++)
{
//int index = i*n+j;
a[i][j] = b[i][j] = (i==j?1.0f:0.0f);
}
}
double *u = (double *) malloc ( n*sizeof(double) );
double *v = (double *) malloc ( n*sizeof(double) );
//initiate u and v
for (i= 1; i< n+1; i++)
{
u[i-1] = 1.0f/(n+1.0f-i);
v[i-1] = sqrt(i);
}
//vTu
double vTu = 0.0f;
for (i= 0; i< n; i++)
{
vTu += u[i]*v[i];
}
double scalar = 1.0f/(1.0f+vTu);
//uvT
for (i= 0; i< n; i++)
{
for (j= 0; j< n; j++)
{
uvT[i*n+j] = u[i]*v[j];
}
}
//construct a and b
for (i=0; i< n; i++)
{
for (j= 0; j< n; j++)
{
int index = i*n+j;
a[i][j] += uvT[index];
b[i][j] -= scalar*uvT[index];
}
}
free (uvT);
free (u);
free (v);
}
//############################################################

//MAIN
int main (int argc, char *argv[]){
	if(argc > 1)
		DIM_N = atoi(argv[1]); // here to enter the size of the matrix
	if(argc > 2)
		threads = atoi(argv[2]); // here to enter the number of the T
	if(argc > 3)
		threshold = atoi(argv[3]); // here to enter the number of the T
	

  double etime=0.0,stime=0.0;
  double dtime=0.0;
  int i,j,k;
  
  A = (double**) malloc(sizeof(double*)*DIM_N);
  for (i = 0; i< DIM_N; i++){
    A[i] = (double*) malloc(sizeof(double)*DIM_N);
  }
  B = (double**) malloc(sizeof(double*)*DIM_N);
  for (i = 0; i< DIM_N; i++){
    B[i] = (double*) malloc(sizeof(double)*DIM_N);
  }
  C = (double**) malloc(sizeof(double*)*DIM_N);
  for (i = 0; i< DIM_N; i++){
    C[i] = (double*) malloc(sizeof(double)*DIM_N);
  }

  CC = (double**) malloc(sizeof(double*)*DIM_N);
  for (i = 0; i< DIM_N; i++){
    CC[i] = (double*) malloc(sizeof(double)*DIM_N);
  }
//printf("double = %d, double = %d, double* = %d, double* = %d\n", sizeof(double), sizeof(double), sizeof(double*), sizeof(double*));
  
 	for (i = 0; i <DIM_N; i++){
	for (j = 0; j <DIM_N; j++){ 
		A[i][j] = (i+j) * 1.0;
		B[i][j] = (i+j) * 1.0;
		C[i][j]= 0;
		CC[i][j]=0;
	} 
	}

	//accuracyTestInit(A, B, DIM_N);
//####################################################
//### Print the A , B matrices #######################
//####################################################
  //print out the result

printf("computing sequential\n");
 stime=omp_get_wtime();
        for(i=0; i<DIM_N; i++)
                for(j=0; j<DIM_N; j++){
                        CC[i][j] = 0;
                        for(k=0; k < DIM_N; k++)
                               CC[i][j] += A[i][k] * B[k][j];
                }

etime=omp_get_wtime();
printf("computed sequential\n");
dtime=etime-stime;
printf("Sequantial Time taken = %0.3f \n", dtime);

  printf("Num Threads = %d\n",threads);
  //start timer
 stime=omp_get_wtime();
  
  //Strassen Multiplication
  strassenMultMatrix(A,B,C,DIM_N,0,0,0,0,0,0);

  //stop timer
 etime=omp_get_wtime();

  //calculate time taken
    dtime=etime-stime;
  printf("Strassen Time taken = %0.3f \n",dtime);
  
  
  /********Triple Loop Multiplication, with OpenMP, for Comparison**********/
  //start timer
/*stime=omp_get_wtime();
  
  #pragma omp parallel shared(A,B,CC,chunk) private(i,j,k) num_threads(threads)
	{
	  //multiplication process
    #pragma omp for schedule(dynamic) nowait
	    for (j = 0; j < DIM_N; j++){
		    for (i = 0; i < DIM_N; i++){
		      CC[i][j] = 0.0;
			    for (k = 0; k < DIM_N; k++)
				    CC[i][j] += A[i][k] * B[k][j];
				}
		  }
	}
  //normalMultMatrix(A,B,C,DIM_N);*/

  //stop timer
etime=omp_get_wtime();
  

 // dtime=etime-stime;
  //printf("Non-Strassen Time taken = %0.3f \n", dtime);
//####################################################################
//####################################################################
stime=omp_get_wtime();

int result = 0;
int xx=0;
        for(i=0; i < DIM_N; i++){
                for(j=0; j < DIM_N; j++)
                        if(fabs(C[i][j]-CC[i][j])>0.0001){
	//printf("(%d, %d) : (%.20f, %.20f)\n", i, j, C[i][j], CC[i][j]);
                                result = 1;
		xx++;
		//break;
                        }
	//printf("\n");
	//if(result == 1) break;
	}

	printf("\n\nPercentage Error =%.3f\n Error cell=%d\n",(double)xx/(DIM_N*DIM_N),xx);
        printf("Test %s\n", (result == 0) ? "Passed" : "Failed");
	checkPracticalErrors(C, CC, DIM_N);

}

//#########################################################################
//###########################################################################
/*****************************************************************************
 * Note that all following functions only deal with square matrices          *
 * where N is divisible by 2.                                                *
 *****************************************************************************/
void addMatrices(double **x, double **y, double **z, int size,int srow1 , int scol1,int srow2,int scol2){
//performs a matrix addition operation, z=x+y
	int i,j;
	#pragma omp parallel shared(x,y,z,srow1,scol1,srow2,scol2,size) private(i,j) num_threads(threads) 
	{
     	#pragma omp for schedule(static) 
	      for (i = 0; i < size; i++)
		      for (j = 0; j < size; j++)
			      z[i][j] = x[i+srow1][j+scol1] + y[i+srow2][j+scol2];  
	}
}
void addMatricesc(double **x, double **y, double **z, int size,int srow1 , int scol1,int srow2,int scol2,int srow3,int scol3){
//performs a matrix addition operation, z=x+y
	int i,j;
	#pragma omp parallel shared(x,y,z,srow1,scol1,srow2,scol2,srow3,scol3,size) private(i,j) num_threads(threads) 
	{
     	#pragma omp for schedule(static) 
	      for (i = srow3; i < size+srow3; i++)
		      for (j = srow3; j < size+scol3; j++)
			      z[i][j] = x[i-srow3+srow1][j-scol3+scol1] + y[i-srow3+srow2][j-scol3+scol2];  
	}
}
void addMatrices1(double **x, double **y, double **z, int size,int srow1 , int scol1,int srow2,int scol2){
//performs a matrix addition operation, z=x+y
	int i,j;
	#pragma omp parallel shared(x,y,z,srow1,scol1,srow2,scol2,size) private(i,j) num_threads(threads) 
	{
     	#pragma omp for schedule(static) 
	      for (i = srow2; i < size+srow2; i++)
		      for (j = scol2; j < size+scol2; j++)
			      z[i][j] = x[i-srow2+srow1][j-scol2+scol1] + y[i-srow2][j-scol2];  
	}
}

void subMatrices(double **x, double **y, double **z, int size , int srow1 , int scol1,int srow2,int scol2){
//performs a matrix subtraction operation, z=x-y
	int i,j;
	#pragma omp parallel shared(x,y,z,srow1,scol1,srow2,scol2,size) private(i,j) num_threads(threads)
	{
     	#pragma omp for schedule(static) 
	      for (i = 0; i < size; i++)
		      for (j = 0; j < size; j++)
			      z[i][j] = x[i+srow1][j+scol1] - y[i+srow2][j+scol2];
	}
}
void subMatricesc(double **x, double **y, double **z, int size , int srow1 , int scol1,int srow2,int scol2,int srow3,int scol3){
//performs a matrix subtraction operation, z=x-y
	int i,j;
	#pragma omp parallel shared(x,y,z,srow1,scol1,srow2,scol2,srow3,scol3,size) private(i,j) num_threads(threads)
	{
     	#pragma omp for schedule(static) 
	      for (i = srow3; i < size+srow3; i++)
		      for (j = scol3; j < size+scol3; j++)
			      z[i][j] = x[i-srow3+srow1][j-scol3+scol1] - y[i-srow3+srow2][j-scol3+scol2];
	}
}
void subMatrices1(double **x, double **y, double **z, int size , int srow1 , int scol1,int srow2,int scol2){
//performs a matrix subtraction operation, z=x-y
	int i,j;
	#pragma omp parallel shared(x,y,z,srow1,scol1,srow2,scol2,size) private(i,j) num_threads(threads)
	{
     	#pragma omp for schedule(static) 
	      for (i = srow2; i < size+srow2; i++)
		      for (j = scol2; j < size+scol2; j++)
			      z[i][j] = x[i-srow2+srow1][j-scol2+scol1] - y[i-srow2][j-scol2];
	}
}



void normalMultMatrix(double **x, double **y, double **z, int size,int srow1 , int scol1,int srow2,int scol2,int srow3,int scol3){
//multiplys two matrices: z=x*y
	int i,j,k;
	
	#pragma omp parallel shared(x,y,z,size,srow1,scol1,srow2,scol2,srow3,scol3) private(i,j,k) num_threads(threads)
	{
	  //multiplication process
    	#pragma omp for schedule(static) 
	    for (i = srow3; i < size+srow3; i++){
		    for (j = scol3; j < size+scol3; j++){
		      	            z[i][j] = 0.0;
			    for (k = 0; k < size; k++)
				    z[i][j] += x[i-srow3+srow1][k+scol1] * y[k+srow2][j-scol3+scol2];
				}
      }
	}
}

void myprint(double **xx,char *name,int size,int x,int y)
{
int i,j;
printf("\nStart of Print %s\n",name);
for(i=0;i<size;i++)
{
   for(j=0;j<size;j++)
	{
		printf("%.5f\t",xx[i+x][j+y]);
	}
	printf("\n");
}
printf("\nEnd of Print\n");
}

void strassenMultMatrix(double **a,double **b,double **c,int size,int srow1, int scol1, int srow2 , int scol2 , int srow3 ,int scol3){
//Performs a Strassen matrix multiply operation

  double **t1, **t2;
  int newsize = size/2;
  int i;
printf("\nindeces=%d %d %d %d %d %d %d\n",size,srow1,scol1,srow2,scol2,srow3,scol3);
  
  if (size >= threshold) {
   
    t1 = (double**) malloc(sizeof(double*)*newsize);
    t2 = (double**) malloc(sizeof(double*)*newsize);
 
    
    for (i=0; i < newsize; i++){   
      t1[i] = (double*) malloc(sizeof(double)*newsize);
      t2[i] = (double*) malloc(sizeof(double)*newsize);
      
    }
    //addMatrices(a11,a22,t1,newsize);
    //addMatrices(b11,b22,t2,newsize);
   // strassenMultMatrix(t1,t2,c21,newsize);
	addMatrices(a,a,t1,newsize,0,0,newsize,newsize);
	addMatrices(b,b,t2,newsize,0,0,newsize,newsize);
	strassenMultMatrix(t1,t2,c,newsize,0,0,0,0,newsize,0);//calculate M1


   // subMatrices(a21,a11,t1,newsize);
	subMatrices(a,a,t1,newsize,newsize,0,0,0);

    //addMatrices(b11,b12,t2,newsize);
	addMatrices(b,b,t2,newsize,0,0,0,newsize);

    //strassenMultMatrix(t1,t2,c22,newsize);
	strassenMultMatrix(t1,t2,c,newsize,0,0,0,0,newsize,newsize);//Calculate M6

//###################################################################################Until here the operation is right
    //subMatrices(a12,a22,t1,newsize);
	subMatrices(a,a,t1,newsize,0,newsize,newsize,newsize);
    //addMatrices(b21,b22,t2,newsize);
	addMatrices(b,b,t2,newsize,newsize,0,newsize,newsize);
    //strassenMultMatrix(t1,t2,c11,newsize);
	strassenMultMatrix(t1,t2,c,newsize,0,0,0,0,0,0);//calculate M7

//################################################################################Until Here the operation is right


// Need to define another addition function to make it possible to done
    //addMatrices(c11,c21,c11,newsize);
	addMatricesc(c,c,c,newsize,0,0,newsize,0,0,0);

    //addMatrices(c21,c22,c22,newsize);
	addMatricesc(c,c,c,newsize,newsize,0,newsize,newsize,newsize,newsize);


//####################################################################################
	//addMatrices(a21,a22,t1,newsize);
	addMatrices(a,a,t1,newsize,newsize,0,newsize,newsize);

	//strassenMultMatrix(t1,b11,c21,newsize);
	strassenMultMatrix(t1,b,c,newsize,0,0,0,0,newsize,0); // Compute M2


	//subMatrices(b12,b22,t2,newsize);
	subMatrices(b,b,t2,newsize,0,newsize,newsize,newsize);
	//strassenMultMatrix(a11,t2,c12,newsize)
	strassenMultMatrix(a,t2,c,newsize,0,0,0,0,0,newsize);//Compute M3


	//subMatrices(c22,c21,c22,newsize);
	subMatricesc(c,c,c,newsize,newsize,newsize,newsize,0,newsize,newsize);
       // addMatrices(c22,c12,c22,newsize);
	addMatricesc(c,c,c,newsize,newsize,newsize,0,newsize,newsize,newsize);


//################################################################

	//subMatrices(b21,b11,t2,newsize);
	subMatrices(b,b,t2,newsize,newsize,0,0,0);

	//strassenMultMatrix(a22,t2,t1,newsize);
	strassenMultMatrix(a,t2,t1,newsize,newsize,newsize,0,0,0,0);//compute M4



        //addMatrices(c11,t1,c11,newsize);
	addMatrices1(c,t1,c,newsize,0,0,0,0);

       // addMatrices(c21,t1,c21,newsize);
	addMatrices1(c,t1,c,newsize,newsize,0,newsize,0);




	//addMatrices(a11,a12,t1,newsize);
	addMatrices(a,a,t1,newsize,0,0,0,newsize);
	//strassenMultMatrix(t1,b22,t2,newsize);
	strassenMultMatrix(t1,b,t2,newsize,0,0,newsize,newsize,0,0);

	//subMatrices(c11,t2,c11,newsize);
	subMatrices1(c,t2,c,newsize,0,0,0,0);

        //addMatrices(c12,t2,c12,newsize);
	addMatrices1(c,t2,c,newsize,0,newsize,0,newsize);

  
       

    free(t1);free(t2);
  }
  else {
    normalMultMatrix(a,b,c,size,srow1,scol1,srow2,scol2,srow3,scol3);
  }
}
