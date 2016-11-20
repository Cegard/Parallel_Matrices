#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <stdlib.h>
#include <stdio.h>

#define MATRIX_DIM 10

__host__ static __inline__ double rand_01()
{
    return (double) (10.0*rand()/(RAND_MAX+1.0));
}


int main(){
    thrust::host_vector<double> h_a(MATRIX_DIM), h_b(MATRIX_DIM), h_c(MATRIX_DIM);
    thrust::generate(h_a.begin(), h_a.end(), rand_01);
    thrust::generate(h_b.begin(), h_b.end(), rand_01);
    
    thrust::device_vector<double> d_a = h_a;
    thrust::device_vector<double> d_b = h_b;
    thrust::device_vector<double> d_c = h_c;
    thrust::transform(d_a.begin(), d_a.end(),
                      d_b.begin(), d_c.begin(),
                      thrust::multiplies<double>());
    thrust::copy(d_c.begin(), d_c.end(), h_c.begin());
    
    for (int i = 0; i < MATRIX_DIM; i++)
        printf("%.2f + %.2f = %.2f \n", h_a[i], h_b[i], h_c[i]);
    return 0;
}