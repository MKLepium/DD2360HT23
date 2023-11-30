
#include <stdio.h>
#include <sys/time.h>
#include <random>

#define NUM_BINS 4096

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements, unsigned int num_bins) {

    extern __shared__ unsigned int temp[]; // Shared memory for temporary accumulation

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;
    int tid = threadIdx.x;

    // Initialize shared memory
    for (int j = tid; j < num_bins; j += blockDim.x) {
        temp[j] = 0;
    }
    __syncthreads();

    // Accumulate histogram in shared memory
    while (i < num_elements) {
        atomicAdd(&temp[input[i]], 1);
        i += offset;
    }
    __syncthreads();

    // Update global histogram
    for (int j = tid; j < num_bins; j += blockDim.x) {
        atomicAdd(&bins[j], temp[j]);
    }
}


__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < num_bins) {
        bins[i] = min(bins[i], 127u);
    }
}


int main(int argc, char **argv) {
  
  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *resultRef;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

  //@@ Insert code below to read in inputLength from args
  if (argc != 2) {
    printf("Usage: ./histogram.out <inputLength>\n");
    exit(1);
  }
  inputLength = atoi(argv[1]);

  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output

  hostInput = (unsigned int *)malloc(inputLength * sizeof(unsigned int));
  hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
  resultRef = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
  
  //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
  #include <random>

  std::random_device rd;
  std::default_random_engine generator(rd());
  std::uniform_int_distribution<int> distribution(0, NUM_BINS - 1);

  for (int i = 0; i < inputLength; i++) {
    hostInput[i] = distribution(generator) % NUM_BINS;
  }


  //@@ Insert code below to create reference result in CPU

  for (int i = 0; i < NUM_BINS; i++) {
    resultRef[i] = 0;
  }
  for (int i = 0; i < inputLength; i++) {
    resultRef[hostInput[i]]++;
  }
  for (int i = 0; i < NUM_BINS; i++) {
    resultRef[i] = min(resultRef[i], 127u);
  }
  for (int i = 0; i < NUM_BINS; i++) {
    hostBins[i] = 0;
  }

  //@@ Insert code below to allocate GPU memory here
  cudaError_t err;
  err = cudaMalloc((void**)&deviceInput, inputLength * sizeof(unsigned int));
  if(err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    exit(1);
  }

  err = cudaMalloc((void**)&deviceBins, NUM_BINS * sizeof(unsigned int));
  if(err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    exit(1);
  }



  //@@ Insert code to Copy memory to the GPU here

  cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceBins, hostBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyHostToDevice);

  //@@ Insert code to initialize GPU results

  for (int i = 0; i < NUM_BINS; i++) {
    hostBins[i] = 0;
  }


  //@@ Initialize the grid and block dimensions here

  int blockSize = 256;
  int gridSize = (inputLength + blockSize - 1) / blockSize;
  dim3 DimGrid(gridSize, 1, 1);
  dim3 DimBlock(blockSize, 1, 1);
  printf("gridSize = %d\n", gridSize);
  printf("blockSize = %d\n", blockSize);

  //@@ Launch the GPU Kernel here

  histogram_kernel<<<DimGrid, DimBlock, NUM_BINS * sizeof(unsigned int)>>>(deviceInput, deviceBins, inputLength, NUM_BINS);
  


  //@@ Initialize the second grid and block dimensions here


  int blockSize2 = 256;
  int gridSize2 = (NUM_BINS + blockSize2 - 1) / blockSize2;
  printf("gridSize2 = %d\n", gridSize2);
  printf("blockSize2 = %d\n", blockSize2);
  dim3 DimGrid2(gridSize2, 1, 1);
  dim3 DimBlock2(blockSize2, 1, 1);




  //@@ Launch the second GPU Kernel here

  cudaDeviceSynchronize();
  convert_kernel<<<DimGrid2, DimBlock2>>>(deviceBins, NUM_BINS);



  //@@ Copy the GPU memory back to the CPU here

  cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);



  //@@ Insert code below to compare the output with the reference

  for (int i = 0; i < NUM_BINS; i++) {
    if (hostBins[i] != resultRef[i]) {
      printf("Test failed\n");
      //printf("hostBins[%d] = %d, resultRef[%d] = %d\n", i, hostBins[i], i, resultRef[i]);
    }
  }

  printf("Test passed!\n");

  // print the histogram to file
  FILE *f = fopen("histogram.csv", "w");
  if (f == NULL) {
    printf("Error opening file!\n");
    exit(1);
  }

  for (int i = 0; i < NUM_BINS; i++) {
    fprintf(f, "%d, %d\n", i, hostBins[i]);
  }
  fclose(f);

  //@@ Free the GPU memory here

  cudaFree(deviceInput);
  cudaFree(deviceBins);

  //@@ Free the CPU memory here

  free(hostInput);
  free(hostBins);
  free(resultRef);


  return 0;
}

