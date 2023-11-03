
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <chrono>


#define DataType float

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < numARows && col < numBColumns){
        DataType sum = 0;
        for(int i = 0; i < numAColumns; i++){
            sum += A[row * numAColumns + i] * B[i * numBColumns + col];
        }
        C[row * numBColumns + col] = sum;
    }
}

int main(int argc, char **argv) {
    
    DataType *hostA; // The A matrix
    DataType *hostB; // The B matrix
    DataType *hostC; // The output C matrix
    DataType *resultRef; // The reference result
    DataType *deviceA;
    DataType *deviceB;
    DataType *deviceC;
    int numARows;    // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows;    // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows;
    int numCColumns;

    //@@ Insert code below to read in numARows, numAColumns, numBColumns and numBRows from args
    if(argc != 4){
        printf("Usage: %s numARows numAColumns numBColumns\n", argv[0]);
        exit(1);
    }
    numARows = atoi(argv[1]);
    numAColumns = atoi(argv[2]);
    numBColumns = atoi(argv[3]);
    numBRows = numAColumns;
    numCRows = numARows;
    numCColumns = numBColumns;
    if(numARows <= 0 || numAColumns <= 0 || numBColumns <= 0){
        printf("Invalid matrix dimension\n");
        exit(1);
    }

    printf("Matrix dimension: %d x %d, %d x %d, %d x %d\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);


    //@@ Insert code below to allocate Host memory for input and output
    hostA = (DataType *)malloc(numARows * numAColumns * sizeof(DataType));
    hostB = (DataType *)malloc(numBRows * numBColumns * sizeof(DataType));
    if(hostA == NULL || hostB == NULL){
        printf("Error allocating memory\n");
        exit(1);
    }


    //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
    // Initialize hostA and hostB to random numbers
    for(int i = 0; i < numARows; i++) {
        for(int j = 0; j < numAColumns; j++) {
            hostA[i * numAColumns + j] = (DataType)rand() / RAND_MAX;
        }
    }

    for(int i = 0; i < numBRows; i++) {
        for(int j = 0; j < numBColumns; j++) {
            hostB[i * numBColumns + j] = (DataType)rand() / RAND_MAX;
        }
    }

    // Allocate memory for resultRef
    resultRef = (DataType *)malloc(numCRows * numCColumns * sizeof(DataType));
    if (resultRef == NULL) {
        printf("Error allocating memory for resultRef\n");
        exit(1);
    }

    // Compute the reference result in CPU
    auto startRef = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < numARows; i++) {
        for(int j = 0; j < numBColumns; j++) {
            resultRef[i * numBColumns + j] = 0; // Initialize the element to 0
            for(int k = 0; k < numAColumns; k++) {
                // Accumulate the sum for the dot product of row i from A and column j from B
                resultRef[i * numBColumns + j] += hostA[i * numAColumns + k] * hostB[k * numBColumns + j];
            }
        }
    }
    auto endRef = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedRef = endRef - startRef;
    printf("Computation made on CPU: %f s\n", elapsedRef.count());

    hostC = (DataType *)malloc(numCRows * numCColumns * sizeof(DataType));
    if(hostA == NULL || hostB == NULL || hostC == NULL || resultRef == NULL){
        printf("Error allocating memory\n");
        exit(1);
    }

    //@@ Insert code below to allocate GPU memory here
    cudaMalloc((void **)&deviceA, numARows * numAColumns * sizeof(DataType));
    cudaMalloc((void **)&deviceB, numBRows * numBColumns * sizeof(DataType));
    cudaMalloc((void **)&deviceC, numCRows * numCColumns * sizeof(DataType));
    
    //@@ Insert code to below to Copy memory to the GPU here
    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(DataType), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(DataType), cudaMemcpyHostToDevice);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    printf("Copy memory to GPU: %f s\n", elapsed.count());
    
    //@@ Initialize the grid and block dimensions here
    dim3 dimGrid((numBColumns - 1) / 32 + 1, (numARows - 1) / 32 + 1, 1);
    dim3 dimBlock(32, 32, 1);
    
    //@@ Launch the GPU Kernel here
    start = std::chrono::high_resolution_clock::now();
    gemm<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    printf("GPU Kernel: %f s\n", elapsed.count());


    //@@ Copy the GPU memory back to the CPU here
    start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(DataType), cudaMemcpyDeviceToHost);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    printf("Copy memory to CPU: %f s\n", elapsed.count());
    



    //@@ Insert code below to compare the output with the reference
    int error = 0;
    for(int i = 0; i < numCRows; i++) {
        for(int j = 0; j < numCColumns; j++) {
            if(abs(hostC[i * numCColumns + j] - resultRef[i * numCColumns + j]) > 1e-2){
                error = 1;
                break;
            }
        }
    }

    if(error) {
        printf("The results are incorrect!\n");
    } else {
        printf("The results are correct.\n");
    }



    //@@ Free the GPU memory here
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);



    //@@ Free the CPU memory here
    free(hostA);
    free(hostB);
    free(hostC);
    free(resultRef);



    return 0;
}
