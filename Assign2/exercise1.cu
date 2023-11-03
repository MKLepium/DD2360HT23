#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

#define DataType double

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < len) {
        out[i] = in1[i] + in2[i];
    }
}

std::chrono::high_resolution_clock::time_point start_timer() {
    return std::chrono::high_resolution_clock::now();
}

double stop_timer(std::chrono::high_resolution_clock::time_point start_time) {
    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end_time - start_time).count();
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
        printf("Usage: %s inputLength\n", argv[0]);
        return 1;
    }
    
    inputLength = atoi(argv[1]);
    printf("The input length is %d\n", inputLength);

    hostInput1 = (DataType *)malloc(sizeof(DataType) * inputLength);
    hostInput2 = (DataType *)malloc(sizeof(DataType) * inputLength);
    hostOutput = (DataType *)malloc(sizeof(DataType) * inputLength);
    resultRef = (DataType *)malloc(sizeof(DataType) * inputLength);

    // Initialize inputs and reference
    for (int i = 0; i < inputLength; i++) {
        hostInput1[i] = static_cast<DataType>(rand()) / static_cast<DataType>(RAND_MAX);
        hostInput2[i] = static_cast<DataType>(rand()) / static_cast<DataType>(RAND_MAX);
        resultRef[i] = hostInput1[i] + hostInput2[i];
    }

    // Allocate device memory
    cudaMalloc((void **)&deviceInput1, sizeof(DataType) * inputLength);
    cudaMalloc((void **)&deviceInput2, sizeof(DataType) * inputLength);
    cudaMalloc((void **)&deviceOutput, sizeof(DataType) * inputLength);

    // Copy data from host to device
    auto start_memcpy_h2d = start_timer();
    cudaMemcpy(deviceInput1, hostInput1, sizeof(DataType) * inputLength, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, hostInput2, sizeof(DataType) * inputLength, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    double time_memcpy_h2d = stop_timer(start_memcpy_h2d);

    // Launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (inputLength + threadsPerBlock - 1) / threadsPerBlock;
    
    auto start_kernel = start_timer();
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
    cudaDeviceSynchronize();
    double time_kernel = stop_timer(start_kernel);

    // Copy data from device back to host
    auto start_memcpy_d2h = start_timer();
    cudaMemcpy(hostOutput, deviceOutput, sizeof(DataType) * inputLength, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    double time_memcpy_d2h = stop_timer(start_memcpy_d2h);

    // Output the timing information
    printf("Time spent copying data host to device: %f ms\n", time_memcpy_h2d);
    printf("Time spent executing kernel: %f ms\n", time_kernel);
    printf("Time spent copying data device to host: %f ms\n", time_memcpy_d2h);

    // Check the results
    int error = 0;
    for (int i = 0; i < inputLength; i++) {
        if (fabs(hostOutput[i] - resultRef[i]) > 1e-5) {
            error = 1;
            break;
        }
    }
    
    if (error == 0) {
        printf("The output is correct\n");
    } else {
        printf("The output is wrong\n");
    }

    // Cleanup
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);
    free(hostInput1);
    free(hostInput2);
    free(hostOutput);
    free(resultRef);

    return 0;
}
