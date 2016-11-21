#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

#define MATRIX_DIM 10


__host__ double* createArrayWithRandoms(){
    double *matrix = (double *) malloc(MATRIX_DIM * sizeof(double));
    
    for (int i = 0; i < MATRIX_DIM; i++)
        *(matrix + i) = (10.0*rand()/(RAND_MAX+1.0));
    
    return matrix;
}


__global__ void addArrays(double *a, double *b, double *c, int length){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < length)
        *(c + i) = *(a + i) + *(b + i);
}


__host__ void checkError(cudaError_t error, const char *point){
    
    if (error != cudaSuccess){
        printf("there was an error at %s, error code: %d", point, error);
        exit(EXIT_FAILURE);
    }
}


int main(){
    
    cudaError_t error = cudaSuccess;
    size_t size = MATRIX_DIM * sizeof(double);
    double *h_a = createArrayWithRandoms();
    double *h_b = createArrayWithRandoms();
    double *h_c = (double *) malloc(MATRIX_DIM * sizeof(double));
    double *d_a , *d_b, *d_c;
    
    error = cudaMalloc(&d_a, size);
    checkError(error, "allocating device memory for A");
    error = cudaMalloc(&d_b, size);
    checkError(error, "allocating device memory for B");
    error = cudaMalloc(&d_c, size);
    checkError(error, "allocating device memory for C");
    
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    checkError(error, "copy A from host to device");
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    checkError(error, "copy B from host to device");
    cudaMemcpy(d_c, h_c, size, cudaMemcpyHostToDevice);
    checkError(error, "copy C from host to device");
    
    addArrays<<< 1, MATRIX_DIM >>>(d_a, d_b, d_c, MATRIX_DIM);
    
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    checkError(error, "copy C from device to host");
    
    for (int i = 0; i < MATRIX_DIM; i++)
        printf("%.2f + %.2f = %.2f \n", *(h_a + i), *(h_b + i), *(h_c + i));
    
    free(h_a);
    free(h_b);
    free(h_c);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}