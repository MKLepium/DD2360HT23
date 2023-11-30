NVCC = nvcc

TARGET = histogram.out

SRC = $(wildcard *.cu)

OBJ = $(SRC:.cu=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(NVCC) -o $@ $^

%.o: %.cu
	$(NVCC) -c $<

clean:
	rm -f $(OBJ) $(TARGET)