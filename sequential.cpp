#include <stdio.h>
#include <stdlib.h>
#include <ctime>

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


double** multiplyMatrix(double **matrixA, double **matrixB, int dim){
    double **matrixC = createMatrixWithZeroes(dim);
    
    for (int i = 0; i < dim*dim; i++){
		double cell = 0.0;
		int row = i/dim;
		int col = i%dim;
		
		for (int j = 0; j < dim; j++)
			cell += *(*(matrixA + row) + j) * *(*(matrixB + j) + col);
		
		*(*(matrixC + row) + col) = cell;
    }
    
    return matrixC;
}


int main(int argc, const char** argv){
	int dim =  atoi(*(argv + 1));
    double **matrixA = createRandomMatrix(dim);
    double **matrixB = createRandomMatrix(dim);
    std::clock_t tic;
    tic = std::clock();
	double **matrixC = multiplyMatrix(matrixA, matrixB, dim);
    double seconds = (double) (std::clock() - tic) / 1000000.0;
	/*
	printf("\n");
    printf("matrix a\n");
    printMatrix(matrixA);
    printf("matrix b\n");
    printMatrix(matrixB);
	printf("matrix c\n");
	printMatrix(matrixC);
	*/
    printf("Taken time for a %dX%d matrix: %.5f\n",dim, dim, seconds);
    freeMatrix(&matrixA, dim);
    freeMatrix(&matrixB, dim);
	freeMatrix(&matrixC, dim);
    
	return 0;
}