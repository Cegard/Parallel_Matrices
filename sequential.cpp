#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

using namespace std; 


double** createMatrixWithZeroes(int dim){
    double **matrix = (double**) malloc(dim * sizeof(double*));
    
    for (int i = 0; i < dim; i++){
        *(matrix + i) = (double*) malloc(dim * sizeof(double));
        
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
    double **matrix = (double**) malloc(dim * sizeof(double*));
    
    for (int i = 0; i < dim; i++){
        *(matrix + i) = (double*) malloc(dim * sizeof(double));
        
        for (int j = 0; j < dim; j++)
            *(*(matrix + i) + j) = (10.0*rand()/(RAND_MAX+1.0));
    }
    
    return matrix;
}


double** multiplyMatrix(double **matrixA, double **matrixB, double **matrixC, int dim){
    double **answer = matrixC;
    
    for (int i = 0; i < dim*dim; i++){
		double cell = 0.0;
		int row = i/dim;
		int col = i%dim;
		
		for (int j = 0; j < dim; j++)
			cell += *(*(matrixA + row) + j) * *(*(matrixB + j) + col);
		
		*(*(matrixC + row) + col) = cell;
    }
    
    return answer;
}


int main(){ //int argc, const char** argv){
	int dim = 32; // atoi(*(argv + 1));
	printf("\n");
	
	while (dim <= 1024){
		double **matrixA = createRandomMatrix(dim);
		double **matrixB = createRandomMatrix(dim);
		double **matrixC = createMatrixWithZeroes(dim);
		struct timeval start_time;
		struct timeval end_time;
		gettimeofday(&start_time, NULL);
		matrixC = multiplyMatrix(matrixA, matrixB, matrixC, dim);
		gettimeofday(&end_time, NULL);
		double seconds = (((1000.0*end_time.tv_sec) + (end_time.tv_usec/1000.0)) -
						 ((1000.0*start_time.tv_sec) + (start_time.tv_usec/1000.0)))/1000.0;
		
		//printf("\n");
		//printf("matrix a\n");
		//printMatrix(matrixA);
		//printf("matrix b\n");
		//printMatrix(matrixB);
		//printf("matrix c\n");
		//printMatrix(matrixC);
		
		printf("Taken time for a %dX%d matrix: %.5f\n",dim, dim, seconds);
		freeMatrix(&matrixA, dim);
		freeMatrix(&matrixB, dim);
		freeMatrix(&matrixC, dim);
		dim *= 2;
	}
	
	printf("\n");
    
	return 0;
}