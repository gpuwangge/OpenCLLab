// Constants for kernels 1 -- 5
#define TS 32                        // The square-root of the 2D tile-size (== work-group dims)

// Constants for kernels 3, 5
#define WPT 8                        // The amount of work-per-thread, i.e. the thread-coarsening factor
#define RTS (TS/WPT)                 // The reduced tile-size in one dimension

// Constants for kernels 4, 7 -- 10
#define WIDTH 4                      // The vector-width (in number of floats)

// Constants for kernel 5
#define TSDK 16                      // The tile-size in dimension K (for kernel 5 only)
#define LPT ((TSDK*WPT)/(TS))        // The amount of loads-per-thread (assume TSN==TSM)

// Constants for kernels 6 -- 10
#define TSM 128                      // The tile-size in dimension M
#define TSN 128                      // The tile-size in dimension N
#define TSK 16                       // The tile-size in dimension K
#define WPTM 8                       // The amount of work-per-thread in dimension M
#define WPTN 8                       // The amount of work-per-thread in dimension N
#define RTSM (TSM/WPTM)              // The reduced tile-size in dimension M (== number of threads)
#define RTSN (TSN/WPTN)              // The reduced tile-size in dimension N (== number of threads)
#define LPTA ((TSK*WPTM*WPTN)/(TSN)) // The amount of loads-per-thread for A
#define LPTB ((TSK*WPTM*WPTN)/(TSM)) // The amount of loads-per-thread for B

kernel void matrixMul1(const int M, const int N, const int K, global const float *A, global const float *B, global float *C ){
    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)

    // Compute a single element (loop over K)
    float acc = 0.0f;
    for (int k=0; k<K; k++) {//column-major multiplication
        acc += A[k*M + globalRow] * B[globalCol*K + k];
    }

    // Store the result
    C[globalCol*M + globalRow] = acc;
}