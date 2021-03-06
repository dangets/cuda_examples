
%.o : %.cu
	nvcc -c $^ $(CFLAGS) $(LDFLAGS)

% : %.cu
	nvcc -o $@  $^ $(CFLAGS) $(LDFLAGS)


BINS = 01_vector_add 02_vector_add_2 03_opengl_ripple


.PHONY: all
all: $(BINS)


02_vector_add_2: 02_vector_add_2.cu CUDATimer.o

03_opengl_ripple: LDFLAGS = -lGL -lglut



.PHONY: clean
clean:
	rm -f *.o
	rm -f $(BINS)

