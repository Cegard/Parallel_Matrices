#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <stdlib.h>
#include <stdio.h>

#define MATRIX_DIM 4


int main(){
    thrust::host_vector<double> h_a(MATRIX_DIM * MATRIX_DIM),
                                h_b(MATRIX_DIM * MATRIX_DIM),
                                h_c(MATRIX_DIM * MATRIX_DIM);
    thrust::device_vector<double> d_a = h_a;
    thrust::device_vector<double> d_b = h_b;
    thrust::device_vector<double> d_c = h_c;
    
    /* Vectors  initiallisation*/
    for (int i = 0; i < MATRIX_DIM; i++){
        
        for (int j = 0; j < MATRIX_DIM; j++){
            h_a[i * MATRIX_DIM + j] = (10.0*rand()/(RAND_MAX+1.0));
            h_b[i * MATRIX_DIM + j] = (10.0*rand()/(RAND_MAX+1.0));
            d_a[i * MATRIX_DIM + j] = h_a[i * MATRIX_DIM + j];
            d_b[j * MATRIX_DIM + i] = h_b[i * MATRIX_DIM + j];
        }
    }
    
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
    
    thrust::copy(d_c.begin(), d_c.end(), h_c.begin());
    
    for (int i = 0; i < MATRIX_DIM; i++){
        
        for(int j = 0; j < MATRIX_DIM; j++)
            printf("%.2f ", h_c[i * MATRIX_DIM + j]);
        
        printf("\n");
    }
    
    return 0;
}