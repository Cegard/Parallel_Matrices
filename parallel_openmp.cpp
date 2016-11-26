#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

#define PAD 2*sizeof(double)

using namespace std; 


double** createMatrixWithZeroes(int dim){
    double **matrix = (double**) malloc(dim * sizeof(double*) * PAD);
    
    for (int i = 0; i < dim; i++){
        *(matrix + i) = (double*) malloc(dim * sizeof(double) + PAD);
        
        for (int j = 0; j < dim; j++)
            *(*(matrix + i) + j) = 0.0;
    }
    
    return matrix;
}


int printMatrix(double **matrix, int dim){
    
    for (int i = 0; i < dim; i++){
        
        for (int j = 0; j < dim; j++)
            printf("%.2f ", *(*(matrix + i) + j));
		
        printf("\n");
    }
    printf("\n");
}


void freeMatrix(double ***matrix, int dim){
    
    for (int i = 0; i < dim; i++)
        free(*(*matrix + i));
    
    free(*matrix);
}


double ** createRandomMatrix(int dim){
    double **matrix = (double**) malloc(dim * sizeof(double*) * PAD);
    
    for (int i = 0; i < dim; i++){
        *(matrix + i) = (double*) malloc(dim * sizeof(double) + PAD);
        
        for (int j = 0; j < dim; j++)
            *(*(matrix + i) + j) = (10.0*rand()/(RAND_MAX+1.0));
    }
    
    return matrix;
}


double** multiplyMatrix(double **matrixA, double **matrixB, double **matrixC, int dim, int threads){
    double **answer = matrixC;
    
	#pragma omp parallel num_threads(threads)
    {
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < dim; i++){
            double cell = 0.0;
            
            for (int j = 0; j < dim; j++){
                
                for (int n = 0; n < dim; n++)
                    cell += *(*(matrixA + i) + j) * *(*(matrixB + j) + n);
                
                *(*(answer + i) + j) = cell;
            }
        }
    }
    
    return answer;
}


int main(){
    int dim = 64;
    int threads = 1;
    printf("\n---------------------------------------------\n");
    
    while (dim <= 1024){
        double **matrixA = createRandomMatrix(dim);
        double **matrixB = createRandomMatrix(dim);
        double **matrixC = createMatrixWithZeroes(dim);
        struct timeval start_time;
        struct timeval end_time;
        gettimeofday(&start_time, NULL);
        matrixC = multiplyMatrix(matrixA, matrixB, matrixC, dim, threads);
        gettimeofday(&end_time, NULL);
        double seconds = (((1000.0*end_time.tv_sec) + (end_time.tv_usec/1000.0)) -
                         ((1000.0*start_time.tv_sec) + (start_time.tv_usec/1000.0)))/1000.0;
        //printf("\n");
        //printf("matrix a\n");
        //printMatrix(matrixA, dim);
        //printf("matrix b\n");
        //printMatrix(matrixB, dim);
        //printf("matrix c\n");
        //printMatrix(matrixC, dim);
        
        printf("Taken time for a matrix of %dX%d with %d threads: %.5fs\n", dim, dim, threads, seconds);
        freeMatrix(&matrixA, dim);
        freeMatrix(&matrixB, dim);
        freeMatrix(&matrixC, dim);
        
        threads *= 2;
        
        if (threads > 32){
            printf("---------------------------------------------\n\n");
            printf("---------------------------------------------\n");
            threads = 1;
            dim *= 2;
        }
        
    }
	return 0;
}