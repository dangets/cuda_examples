#ifndef CUDATIMER_CUH
#define CUDATIMER_CUH

/**
 * Utility class wrapping cuda event timers
 * (less flexible but more convenient)
 *
 * Danny George 2012
 */

class CUDATimer {
    public:
        CUDATimer();
        ~CUDATimer();

        void start(cudaStream_t stream=0);
        float get_elapsed_time_sync(cudaStream_t stream=0);
        float get_elapsed_time_nosync(cudaStream_t stream=0);

    private:
        cudaEvent_t ev_start;
        cudaEvent_t ev_stop;
};

#endif /* end of include guard: CUDATIMER_CUH */

