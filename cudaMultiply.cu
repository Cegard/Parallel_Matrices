#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#define PAD 2*sizeof(double)

using namespace std;


__host__ double* createMatrix(int dim, double zero){
    double *matrix = (double*) malloc(dim * dim * (sizeof(double)));// * PAD));
    
    for (int i = 0; i < dim*dim; i++)
        *(matrix + i) = (zero)? 0.0 : (10.0*rand()/(RAND_MAX+1.0));
    
    return matrix;
}


__host__ double* createMatrixWithZeroes(int dim){
    return createMatrix(dim, 1);
}


__host__ double* createRandomMatrix(int dim){
    return createMatrix(dim, 0);
}


__host__ void printMatrix(double *matrix, int dim){
    
    for (int i = 0; i < dim*dim; i++){
        printf("%.2f ", *(matrix + i));
        
        if (((i + 1) % dim) == 0 )
            printf("\n");
    }
    printf("\n");
}


__global__ void multiplyMatrix(double *matrixA, double *matrixB, double *matrixC,
                               int dim, int threads, int limit){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < limit){
        double cell = 0.0;
        for (int n = 0; n < dim; n++){
        printf("%d %d\n", i, n);
            int row = dim*(i/dim) + n;
            int col = dim*(i%dim) + n;
            cell += *(matrixA + row) * *(matrixB + col);
        }
        
        *(matrixC + i) = cell;
    }
}


int main(){
    int dim =  3;
    int threads = 2;
    printf("\n---------------------------------------------\n");
    while (dim <= 3){
        int limit = dim*dim;
        int size = dim * dim * sizeof(double);// * PAD;
        int blocks = 3;
        int threadsPerBlock = 3;// + (limit/blocks);
        
        double *h_matrixA = createRandomMatrix(dim);
        double *h_matrixB = createRandomMatrix(dim);
        double *h_matrixC = createMatrixWithZeroes(dim);
        double *d_matrixA, *d_matrixB, *d_matrixC;
        
        cudaMalloc(&d_matrixA, size);
        cudaMalloc(&d_matrixB, size);
        cudaMalloc(&d_matrixC, size);
        
        cudaMemcpy(d_matrixA, h_matrixA, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_matrixB, h_matrixB, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_matrixC, h_matrixC, size, cudaMemcpyHostToDevice);
        
        struct timeval start_time;
        struct timeval end_time;
        gettimeofday(&start_time, NULL);
        multiplyMatrix<<<blocks, threadsPerBlock>>>(d_matrixA, d_matrixB, d_matrixC,
                                                    dim, threads, limit);
        gettimeofday(&end_time, NULL);
        double seconds = (((1000.0*end_time.tv_sec) + (end_time.tv_usec/1000.0)) -
                         ((1000.0*start_time.tv_sec) + (start_time.tv_usec/1000.0)))/1000.0;
        cudaMemcpy(h_matrixC, d_matrixC, size, cudaMemcpyDeviceToHost);
        printf("\n");
        printf("matrix a\n");
        printMatrix(h_matrixA, dim);
        printf("matrix b\n");
        printMatrix(h_matrixB, dim);
        printf("matrix c\n");
        printMatrix(h_matrixC, dim);
        printf("Taken time for a matrix of %dX%d with %d threads: %.5fs\n", dim, dim, threads, seconds);
        cudaFree(d_matrixA);
        cudaFree(d_matrixB);
        cudaFree(d_matrixC);
        free(h_matrixA);
        free(h_matrixB);
        free(h_matrixC);
        threads *= 2;
        
        if (threads > 1){
            printf("---------------------------------------------\n\n");
            printf("---------------------------------------------\n");
            threads = 1;
            dim *= 2;
        }
    }
    
	return 0;
}