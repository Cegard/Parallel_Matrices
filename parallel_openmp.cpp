#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <omp.h>

#define PAD 2*sizeof(double)

using namespace std;


double* createMatrix(int dim, double zero){
    double *matrix = (double*) malloc(dim * dim * (sizeof(double) * PAD));
    
    for (int i = 0; i < dim*dim; i++)
        *(matrix + i) = (zero)? 0.0 : (10.0*rand()/(RAND_MAX+1.0));
    
    return matrix;
}


double* createMatrixWithZeroes(int dim){
    return createMatrix(dim, 1);
}


double* createRandomMatrix(int dim){
    return createMatrix(dim, 0);
}


int printMatrix(double *matrix, int dim){
    
    for (int i = 0; i < dim*dim; i++){
        printf("%.2f ", *(matrix + i));
        
        if (((i + 1) % dim) == 0 )
            printf("\n");
    }
    printf("\n");
}


double* multiplyMatrix(double *matrixA, double *matrixB, double *matrixC, int dim, int threads){
    double *answer = matrixC;
    
    #pragma omp parallel num_threads(threads)
    {
        int dim_2 = dim * dim;
        int limit = dim_2 * dim;
        int tileIndex, row_row, row_col, col_row, col_col, rowIndex, colIndex;
        
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < limit; i++){
            tileIndex = i / dim;
            row_row = i / dim_2;
            row_col = i % dim;
            rowIndex = row_row * dim + row_col;
            col_row = row_col;
            col_col = tileIndex % dim;
            colIndex = col_row * dim + col_col;
            *(answer + tileIndex) += *(matrixA + rowIndex) * *(matrixB + colIndex);
        }
    }
    
    return answer;
}


int main(){
    int dim =  64;
    int threads = 2;
    printf("\n---------------------------------------------\n");
    
    while (dim <= 1024){
        double *matrixA = createRandomMatrix(dim);
        double *matrixB = createRandomMatrix(dim);
        double *matrixC = createMatrixWithZeroes(dim);
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
        free(matrixA);
        free(matrixB);
        free(matrixC);
        threads *= 2;
        
        if (threads > 32){
            printf("---------------------------------------------\n\n");
            printf("---------------------------------------------\n");
            threads = 2;
            dim *= 2;
        }
    }
    
	return 0;
}