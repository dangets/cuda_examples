
%.o : %.cu
	nvcc -c $^ $(CFLAGS) $(LDFLAGS)

% : %.cu
	nvcc -o $@  $^ $(CFLAGS) $(LDFLAGS)


BINS = 01_intro 02_vector_add 03_functors


.PHONY: all
all: $(BINS)


03_functors: 03_functors.cu CUDATimer.o


.PHONY: clean
clean:
	rm -f *.o
	rm -f $(BINS)


