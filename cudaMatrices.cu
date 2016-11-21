#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <stdlib.h>
#include <stdio.h>

#define MATRIX_DIM 4


__device__
void get_cell (thrust::device_vector<double> A,
               thrust::device_vector<double> B,
               thrust::device_vector<double> C,
               int index_A, int index_B. int index_C){
    C[index_C] = A[index_A] * B[index_B];
}


__global__
void multiply_matrices(thrust::device_vector<double> A,
                       thrust::device_vector<double> B, int length){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < length){
        
    }
    
}


int main(){
    thrust::host_vector<double> h_a(MATRIX_DIM * MATRIX_DIM),
                                h_b(MATRIX_DIM * MATRIX_DIM),
                                h_c(MATRIX_DIM * MATRIX_DIM);
    thrust::device_vector<double> d_a = h_a;
    thrust::device_vector<double> d_b = h_b;
    thrust::device_vector<double> d_c = h_c;
    
    /* vectors  initiallisation */
    for (int i = 0; i < MATRIX_DIM; i++){
        
        for (int j = 0; j < MATRIX_DIM; j++){
            h_a[i * MATRIX_DIM + j] = (10.0*rand()/(RAND_MAX+1.0));
            h_b[i * MATRIX_DIM + j] = (10.0*rand()/(RAND_MAX+1.0));
            d_a[i * MATRIX_DIM + j] = h_a[i * MATRIX_DIM + j];
            d_b[j * MATRIX_DIM + i] = h_b[i * MATRIX_DIM + j];
        }
    }

    /* matrices multiplication 
    for (int i = 0; i < MATRIX_DIM; i ++){
        
        for (int j = 0; j < MATRIX_DIM; j++){
            thrust::device_vector<double> row_times_col(MATRIX_DIM);
            thrust::transform(d_a.begin() + (i * MATRIX_DIM),
                              d_a.begin() + (i * MATRIX_DIM) + MATRIX_DIM,
                              d_b.begin() + (j * MATRIX_DIM), row_times_col.begin(),
                              thrust::multiplies<double>());
            d_c[i * MATRIX_DIM + j] = thrust::reduce(row_times_col.begin(),
                                                     row_times_col.end(), (double) 0.0,
                                                     thrust::plus<double>());
        }
    }
    */
    
    for (int i = 0; i < MATRIX_DIM; i++){
        
    }
    
    thrust::copy(d_c.begin(), d_c.end(), h_c.begin());
    
    return 0;
}