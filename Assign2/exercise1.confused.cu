
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define DataType double

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < len) {
        out[i] = in1[i] + in2[i];
    }
}

int main(int argc, char **argv) {

    int inputLength;
    DataType *hostInput1;
    DataType *hostInput2;
    DataType *hostOutput;
    DataType *resultRef;
    DataType *deviceInput1;
    DataType *deviceInput2;
    DataType *deviceOutput;

    if (argc != 2) {
        printf("The input length is wrong\n");
        printf("Usage: %s inputLength\n", argv[0]);
        return 1;
    }
    inputLength = atoi(argv[1]);


    printf("The input length is %d\n", inputLength);

    hostInput1 = (DataType *) malloc(sizeof(DataType) * inputLength);
    hostInput2 = (DataType *) malloc(sizeof(DataType) * inputLength);
    hostOutput = (DataType *) malloc(sizeof(DataType) * inputLength);
    resultRef = (DataType *) malloc(sizeof(DataType) * inputLength);

    for (int i = 0; i < inputLength; i++) {
        hostInput1[i] = rand() / (DataType) RAND_MAX;
        hostInput2[i] = rand() / (DataType) RAND_MAX;
        resultRef[i] = hostInput1[i] + hostInput2[i];
    }

    cudaMalloc((void **) &deviceInput1, sizeof(DataType) * inputLength);
    cudaMalloc((void **) &deviceInput2, sizeof(DataType) * inputLength);
    cudaMalloc((void **) &deviceOutput, sizeof(DataType) * inputLength);

    cudaEvent_t start, stop;
    float time_memcpy_h2d, time_memcpy_d2h, time_kernel;
    // Copy data from host (CPU) to device (GPU)
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    cudaMemcpy(deviceInput1, hostInput1, sizeof(DataType) * inputLength, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, hostInput2, sizeof(DataType) * inputLength, cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_memcpy_h2d, start, stop);


    int threadsPerBlock = 256;
    int blocksPerGrid = (inputLength + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventRecord(start, 0);
    // Launch the kernel
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_kernel, start, stop);


    // Copy data from device (GPU) to host (CPU)
    cudaEventRecord(start, 0);
    cudaMemcpy(hostOutput, deviceOutput, sizeof(DataType) * inputLength, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    
    
    cudaEventElapsedTime(&time_memcpy_d2h, start, stop);

    printf("Time spent copying data host to device: %f ms\n", time_memcpy_h2d);
    printf("Time spent executing kernel: %f ms\n", time_kernel);
    printf("Time spent copying data device to host: %f ms\n", time_memcpy_d2h);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    int error = 0;
    for (int i = 0; i < inputLength; i++) {
        if (abs(hostOutput[i] - resultRef[i]) > 1e-5) {
            error = 1;
            break;
        }
    }
    if (error == 0) {
        printf("The output is correct\n");
    } else {
        printf("The output is wrong\n");
    }

    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);
    free(hostInput1);
    free(hostInput2);
    free(hostOutput);
    free(resultRef);


    return 0;
}
