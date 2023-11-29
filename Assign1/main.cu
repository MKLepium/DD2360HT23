#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void addArrays(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 1024; // Number of elements in arrays
    size_t size = n * sizeof(float);


    float *h_a, *h_b, *h_c;
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);


    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }
    std::cout << "h_a: " << h_a[0] << std::endl;
    std::cout << "h_b: " << h_b[0] << std::endl;




    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);


    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);


    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    addArrays<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify the results
    for (int i = 0; i < n; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            std::cout << "Error: " << h_c[i] << " != " << h_a[i] << " + " << h_b[i] << std::endl;
            break;
        }
    }

    std::cout << "Success!" << std::endl;


    // Clean up
    //free(h_a);
    //free(h_b);
    //free(h_c);
    //cudaFree(d_a);
    //cudaFree(d_b);
    //cudaFree(d_c);
    cudaDeviceReset();
    return 0;
}
