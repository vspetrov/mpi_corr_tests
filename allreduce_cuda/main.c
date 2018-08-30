#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

#define MSGLEN_MIN 4096
#define MSGLEN_MAX 4194304

#define COUNT (MSGLEN/sizeof(float))
struct {
    float *sbuf;
    float *rbuf;
    float *cu_sbuf;
    float *cu_rbuf;
    float check;
    cudaStream_t stream;
} global;

#define D_D cudaMemcpyDeviceToDevice
#define D_H cudaMemcpyDeviceToHost
#define H_D cudaMemcpyHostToDevice
#define H_H cudaMemcpyHostToHost

static inline do_memcpy(void *dest, void *src, size_t count, enum cudaMemcpyKind kind) {
    cudaMemcpyAsync(dest, src, count, kind, global.stream);
    cudaStreamSynchronize(global.stream);
}
static void do_test(int count, int is_cuda, int is_inplace) {
    int i;
    float *sbuf = is_inplace ? MPI_IN_PLACE :
        (is_cuda ? global.cu_sbuf : global.sbuf);
    float *rbuf = is_cuda ? global.cu_rbuf : global.rbuf;
    if (is_inplace) {
        if (is_cuda) {
            do_memcpy(rbuf, global.cu_sbuf, count*sizeof(float), D_D);
        } else {
            do_memcpy(rbuf, global.sbuf, count*sizeof(float), H_H);
        }
    } else {
        for (i=0; i<count; i++) {
            global.rbuf[i] = 0.0;
        }
        if (is_cuda) {
            do_memcpy(rbuf, global.rbuf, count*sizeof(float), H_D);
        }
    }

    MPI_Allreduce(sbuf,rbuf,count,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
    if (is_cuda) {
        do_memcpy(global.rbuf, rbuf, count*sizeof(float), D_H);
    }

    for (i=0; i<count; i++) {
        if (fabs(global.rbuf[i] - global.check) > 0.01*global.check) {
            fprintf(stderr, "ERROR: pos %d, value %g, expected %g, diff %g\n",
                    i, global.rbuf[i], global.check, fabs(global.rbuf[i] - global.check));
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }

}

static void run_tests(int *counts, int iters, int rank, int is_cuda, int is_inplace, const char *run_desc) {
    int i;
    if (0 == rank) {
        printf("\n=====================================================\n");fflush(stdout);
        printf("TESTING: %s\n", run_desc);fflush(stdout);
        usleep(10000);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (i=0; i<iters; i++) {
        do_test(counts[i], is_cuda, is_inplace);
        if (0 == rank) {
            if (i/100*100 == i) {
                fprintf(stdout, "progress iter %d\n",i);fflush(stdout);
            }
        }
    }    
}
int main(int argc, char **argv) {
    int rank, size;
    int iters = argc > 1 ? atoi(argv[1]) : 1;
    int seed  = argc > 2 ? atoi(argv[2]) : -1;
    int i, j, status;
    int *counts;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &status);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    counts = (int*)malloc(iters*sizeof(int));
    if (0 == rank) {
        if (seed < 0) {
            srand((int)time(NULL));
        } else {
            srand(seed);
        }
        for (i=0; i<iters; i++) {
            counts[i] = (MSGLEN_MIN + (rand() % (MSGLEN_MAX-MSGLEN_MIN))) / sizeof(float);
        }
    }
    MPI_Bcast(counts, iters, MPI_INT, 0, MPI_COMM_WORLD);

    global.sbuf = (float*)malloc(MSGLEN_MAX);
    global.rbuf = (float*)malloc(MSGLEN_MAX);

    for (j=0; j<MSGLEN_MAX/sizeof(float); j++) {
        global.sbuf[j] = 3.1415*(rank+1);
    }

    global.check = size*(size+1)/2*3.1415;

    cudaStreamCreate(&global.stream);
    cudaMalloc((void**)&global.cu_sbuf, MSGLEN_MAX);
    cudaMalloc((void**)&global.cu_rbuf, MSGLEN_MAX);

    do_memcpy(global.cu_sbuf, global.sbuf, MSGLEN_MAX, H_D);

    run_tests(counts, iters, rank, 0, 0, "HOST");
    run_tests(counts, iters, rank, 0, 1, "HOST_INPLACE");
    run_tests(counts, iters, rank, 1, 0, "CUDA");
    run_tests(counts, iters, rank, 1, 1, "CUDA_INPLACE");

    cudaStreamDestroy(global.stream);
    cudaFree(global.cu_sbuf);
    cudaFree(global.cu_rbuf);
    free(global.sbuf);
    free(global.rbuf);
    MPI_Finalize();
    if (0 == rank) {
        printf("ALL DONE\n");
    }
    return 0;
}
