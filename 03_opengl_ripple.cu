/**
 * An example showing CUDA & OpenGL interoperability heavy copied from
 * CUDA By Example by Jason Sanders and Edward Kandrot
 *
 *
 * There is quite a bit of OpenGL overhead, but the basic idea is that
 * OpenGL needs to manage the memory on the GPU, and CUDA just gets
 * a pointer to that memory to manipulate.
 * The CUDA kernel code is called on each frame draw
 *
 *
 * Besides the OpenGL stuff, this code also demonstrates using more than 1D
 * gridDim and blockDim in the kernel launch parameters.
 * Dimension logic is shown to convert between CUDA thread dimensions to a
 * 2D picture pixel position to a 1D buffer index.
 *
 * Danny George 2012
 */

#define GL_GLEXT_PROTOTYPES

#include <stdio.h>

#include "GL/glut.h"
#include "cuda.h"
#include "cuda_gl_interop.h"


#define DIM     512     // keep as power of 2 above 16


static void HandleError( cudaError_t err, const char *file, int line )
{
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


GLuint bufferObj;
cudaGraphicsResource *resource;


// based on ripple code, but uses uchar4 which is the type of data graphic interop uses
__global__ void kernel(uchar4 *ptr, int ticks)
{
    // map from threadIdx / blockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // map from pixel position to buffer index
    int offset = x + y * blockDim.x * gridDim.x;

    // now calculate the value at that position
    float fx = x - DIM/2;
    float fy = y - DIM/2;
    float d = sqrtf(fx * fx + fy * fy);

    unsigned char grey = (unsigned char)(128.0f + 127.0f *
            cos(d/10.0f - ticks/7.0f) / (d/10.0f + 1.0f));

    ptr[offset].x = grey;   // R
    ptr[offset].y = grey;   // G
    ptr[offset].z = grey;   // B
    ptr[offset].w = 255;    // A
}


static void draw_func(void)
{
    static int ticks = 1;

    // create a devPtr that we can pass to our CUDA kernels
    uchar4 * devPtr;
    size_t size;
    HANDLE_ERROR( cudaGraphicsMapResources(1, &resource, NULL) );
    HANDLE_ERROR( cudaGraphicsResourceGetMappedPointer((void **)&devPtr, &size, resource) );

    dim3 grids(DIM/16, DIM/16);
    dim3 threads(16, 16);

    kernel<<<grids, threads>>>(devPtr, ticks++);

    HANDLE_ERROR( cudaGraphicsUnmapResources(1, &resource, NULL) );

    // pixel buffer is already bound (GL_PIXEL_UNPACK_BUFFER_ARB)
    glDrawPixels(DIM, DIM, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glutSwapBuffers();
}


static void key_func(unsigned char key, int x, int y)
{
    switch (key) {
        case 27:    // ESC
            // clean up OpenGL and CUDA
            HANDLE_ERROR( cudaGraphicsUnregisterResource(resource) );
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
            glDeleteBuffers(1, &bufferObj);
            exit(0);
    }
}



int main(int argc, char *argv[])
{
    cudaDeviceProp prop;
    int dev;

    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 1;
    prop.minor = 0;

    // grab a CUDA device >= 1.0
    //  we need the device number to tell CUDA runtime we
    //  intend to run CUDA & OpenGL on it
    HANDLE_ERROR( cudaChooseDevice(&dev, &prop) );

    HANDLE_ERROR( cudaGLSetGLDevice(dev) );

    // initialize GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(DIM, DIM);
    glutCreateWindow(argv[0]);

    // creating a pixel buffer object (pbo)
    glGenBuffers(1, &bufferObj);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, DIM * DIM * 4,
                 NULL, GL_DYNAMIC_DRAW_ARB);

    // register bufferObj with CUDA runtime as a graphics resource
    HANDLE_ERROR( cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone) );

    // setup GLUT and kick off main loop
    glutKeyboardFunc(key_func);
    glutDisplayFunc(draw_func);
    glutIdleFunc(draw_func);
    glutMainLoop();

    return 0;
}
