#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define NUM_THREADS 16
#define MATRIX_DIM 8

using namespace std; 


double** createMatrixWithZeroes(){
    double **matrix = (double**) malloc(MATRIX_DIM * sizeof(double*));
    
    for (int i = 0; i < MATRIX_DIM; i++){
        *(matrix + i) = (double*) malloc(MATRIX_DIM * sizeof(double));
        
        for (int j = 0; j < MATRIX_DIM; j++)
            *(*(matrix + i) + j) = 0.0;
    }
    
    return matrix;
}


int printMatrix(double **matrix){
    
    for (int i = 0; i < MATRIX_DIM; i++){
        
        for (int j = 0; j < MATRIX_DIM; j++)
            printf("%.2f ", *(*(matrix + i) + j));
		
        printf("\n");
    }
    printf("\n");
}


void freeMatrix(double ***matrix){
    
    for (int i = 0; i < MATRIX_DIM; i++)
        free(*(*matrix + i));
    
    free(*matrix);
}


double ** createRandomMatrix(){
    double **matrix = (double**) malloc(MATRIX_DIM * sizeof(double*));
    
    for (int i = 0; i < MATRIX_DIM; i++){
        *(matrix + i) = (double*) malloc(MATRIX_DIM * sizeof(double));
        
        for (int j = 0; j < MATRIX_DIM; j++)
            *(*(matrix + i) + j) = (10.0*rand()/(RAND_MAX+1.0));
    }
    
    return matrix;
}


double** multiplyMatrix(double **matrixA, double **matrixB){
    double **matrixC = createMatrixWithZeroes();
    
	#pragma omp parallel for
    for (int i = 0; i < MATRIX_DIM*MATRIX_DIM; i++){
		double cell = 0.0;
		int row = i/MATRIX_DIM;
		int col = i%MATRIX_DIM;
		
		#pragma omp parallel for reduction (+:cell)
		for (int j = 0; j < MATRIX_DIM; j++)
			cell += *(*(matrixA + row) + j) * *(*(matrixB + j) + col);
		
		*(*(matrixC + row) + col) = cell;
    }
    
    return matrixC;
}


int main(){
    double **matrixA = createRandomMatrix();
    double **matrixB = createRandomMatrix();
	double **matrixC = multiplyMatrix(matrixA, matrixB);
	
	printf("\n");
    printf("matrix a\n");
    printMatrix(matrixA);
    printf("matrix b\n");
    printMatrix(matrixB);
	printf("matrix c\n");
	printMatrix(matrixC);
	
    freeMatrix(&matrixA);
    freeMatrix(&matrixB);
	freeMatrix(&matrixC);
    
	return 0;
}