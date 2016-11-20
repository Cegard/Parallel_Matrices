#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <stdlib.h>
#include <stdio.h>

#define MATRIX_DIM 4

__host__ static __inline__ double rand_01()
{
    return (double) (10.0*rand()/(RAND_MAX+1.0));
}


int main(){
    thrust::host_vector<double> h_a[MATRIX_DIM], h_b[MATRIX_DIM], h_c[MATRIX_DIM];
    thrust::device_vector<double> d_a[MATRIX_DIM], d_b[MATRIX_DIM], d_c[MATRIX_DIM];
    
    for (int i = 0; i < MATRIX_DIM; i++){
        h_a[i] = thrust::host_vector<double>(MATRIX_DIM);
        h_b[i] = thrust::host_vector<double>(MATRIX_DIM);
        h_c[i] = thrust::host_vector<double>(MATRIX_DIM, 0.0);
        d_c[i] = thrust::host_vector<double>(MATRIX_DIM, 0.0);
        
        thrust::generate(h_a[i].begin(), h_a[i].end(), rand_01);
        thrust::generate(h_b[i].begin(), h_b[i].end(), rand_01);
        
        d_a[i] = h_a[i];
        d_b[i] = h_b[i];
    }
    
    /*
    
    thrust::transform(d_a.begin(), d_a.end(),
                      d_b.begin(), d_c.begin(),
                      thrust::multiplies<double>());
    thrust::copy(d_c.begin(), d_c.end(), h_c.begin());
    */
    
    for(int i = 0; i < MATRIX_DIM; i++){
        
        for(int j = 0; j < MATRIX_DIM; j++)
            printf("%.2f ", h_a[i][j]);
        
        printf("\n");
    }
    return 0;
}