#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <ctime>

#define PAD 64*8

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


double** multiplyMatrix(double **matrixA, double **matrixB, int dim, int threads){
    double **matrixC = createMatrixWithZeroes(dim);
    
	#pragma omp parallel num_threads(threads)
    {
        #pragma omp parallel for
        for (int i = 0; i < dim*dim; i++){
            double cell = 0.0;
            int row = i/dim;
            int col = i%dim;
            
            //#pragma omp parallel for reduction (+:cell)
            for (int j = 0; j < dim; j++)
                cell += *(*(matrixA + row) + j) * *(*(matrixB + j) + col);
            
            *(*(matrixC + row) + col) = cell;
        }
    }
    
    return matrixC;
}


int main(int argc, const char** argv){
	int dim =  atoi(*(argv + 1));
    int threads = atoi(*(argv + 2));
    double **matrixA = createRandomMatrix(dim);
    double **matrixB = createRandomMatrix(dim);
    std::clock_t tic;
    tic = std::clock();
	double **matrixC = multiplyMatrix(matrixA, matrixB, dim, threads);
    double seconds = (double) (std::clock() - tic) / 1000000.0;
    /*
	printf("\n");
    printf("matrix a\n");
    printMatrix(matrixA, dim);
    printf("matrix b\n");
    printMatrix(matrixB, dim);
	printf("matrix c\n");
	printMatrix(matrixC, dim);
	*/
    printf("taken time for a matrix of %dX%d with %d threads: %.5f s\n", dim, dim, threads, seconds);
    freeMatrix(&matrixA, dim);
    freeMatrix(&matrixB, dim);
	freeMatrix(&matrixC, dim);
    
	return 0;
}