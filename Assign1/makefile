# CUDA Compiler
NVCC = nvcc

# Compiler Flags
CFLAGS = -std=c++11

# Target Executable
TARGET = hello_cuda

# Source Files
SOURCES = main.cu

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(NVCC) $(CFLAGS) -o $@ $^

clean:
	rm -f $(TARGET)
