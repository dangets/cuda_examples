
.DEFAULT : all
.PHONY : clean

%.o : %.cu
	nvcc -c $^ $(CFLAGS) $(LDFLAGS)

% : %.cu
	nvcc -o $@  $^ $(CFLAGS) $(LDFLAGS)




BINS :=

01_vector_add: 01_vector_add.cu
BINS += 01_vector_add

02_vector_add_2: 02_vector_add_2.cu CUDATimer.o
BINS += 02_vector_add_2

03_opengl_ripple: LDFLAGS = -lGL -lglut
03_opengl_ripple: 03_opengl_ripple.cu
BINS += 03_opengl_ripple


all: $(BINS)

clean:
	rm -f *.o
	rm -f $(BINS)
