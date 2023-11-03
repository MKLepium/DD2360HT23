# CUDA Compiler
NVCC = nvcc

# Compiler Flags
CFLAGS = -std=c++11

# Target Executable
TARGET = hello_cuda

# Source Files
SOURCES = exercise2.cu

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(NVCC) $(CFLAGS) -o $@.out $^

clean:
	rm -f $(TARGET)
