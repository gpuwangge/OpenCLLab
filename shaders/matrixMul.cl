// Constants for kernels 1 -- 5
#define TS 32                        // The square-root of the 2D tile-size (== work-group dims)
//#define TS 16     //for Iris GPU

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

// Macros for host and kernel code
#define MIN(a,b) ((a) > (b)) ? (b) : (a)
#define MAX(a,b) ((a) > (b)) ? (a) : (b)
#define CEIL_DIV(x,y) (((x) + (y) - 1) / (y))
#define MOD2(x,y) ((x) % (y))
#define DIV2(x,y) ((x) / (y))

// Constants for the supporting transpose kernel
#define TRANSPOSEX 16
#define TRANSPOSEY 16


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

// Tiled and coalesced version
kernel void matrixMul2(const int M, const int N, const int K, global const float *A, global const float *B, global float *C ){
    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)

    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    // Initialise the accumulation register
    float acc = 0.0f;
    
    // Loop over all tiles
    const int numTiles = K/TS;
    for (int t=0; t<numTiles; t++) {

        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        Asub[col][row] = A[tiledCol*M + globalRow];
        Bsub[col][row] = B[globalCol*K + tiledRow];

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            acc += Asub[k][row] * Bsub[col][k];
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final result in C
    C[globalCol*M + globalRow] = acc;
}

// Increased the amount of work-per-thread by a factor WPT
kernel void matrixMul3(const int M, const int N, const int K, global const float *A, global const float *B, global float *C ){
    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS/WPT == RTS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)

    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    // Initialise the accumulation registers
    float acc[WPT];
    for (int w=0; w<WPT; w++) {
        acc[w] = 0.0f;
    }
    
    // Loop over all tiles
    const int numTiles = K/TS;
    for (int t=0; t<numTiles; t++) {

        // Load one tile of A and B into local memory
        for (int w=0; w<WPT; w++) {
            const int tiledRow = TS*t + row;
            const int tiledCol = TS*t + col;
            Asub[col + w*RTS][row] = A[(tiledCol + w*RTS)*M + globalRow];
            Bsub[col + w*RTS][row] = B[(globalCol + w*RTS)*K + tiledRow];
        }

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            for (int w=0; w<WPT; w++) {
                acc[w] += Asub[k][row] * Bsub[col + w*RTS][k];
            }
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final results in C
    for (int w=0; w<WPT; w++) {
        C[(globalCol + w*RTS)*M + globalRow] = acc[w];
    }
}


// Use wider data types
/*
kernel void matrixMul4(const int M, const int N, const int K, global const float *A, global const float *B, global float *C ){

    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS/WIDTH)
    const int col = get_local_id(1); // Local col ID (max: TS)
    const int globalRow = (TS/WIDTH)*get_group_id(0) + row; // Row ID of C (0..M/WIDTH)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)

    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TS][TS/WIDTH];
    __local float Bsub[TS][TS/WIDTH];

    // Initialise the accumulation registers
    #if WIDTH == 1
        float acc = 0.0f;
    #elif WIDTH == 2
        float acc = { 0.0f, 0.0f };
    #elif WIDTH == 4
        float acc = { 0.0f, 0.0f, 0.0f, 0.0f };
    #elif WIDTH == 8
        float acc = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    #endif
    
    // Loop over all tiles
    const int numTiles = K/TS;
    for (int tile=0; tile<numTiles; tile++) {

        // Load one tile of A and B into local memory
        const int tiledRow = (TS/WIDTH)*tile + row;
        const int tiledCol = TS*tile + col;
        Asub[col][row] = A[tiledCol*(M/WIDTH) + globalRow];
        Bsub[col][row] = B[globalCol*(K/WIDTH) + tiledRow];

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the computation for a single tile
        float vecA, vecB;
        float valB;
        for (int k=0; k<TS/WIDTH; k++) {
            vecB = Bsub[col][k];
            for (int w=0; w<WIDTH; w++) {
                vecA = Asub[WIDTH*k + w][row];
                #if WIDTH == 1
                    valB = vecB;
                    acc += vecA * valB;
                #elif WIDTH == 2
                    switch (w) {
                        case 0: valB = vecB.x; break;
                        case 1: valB = vecB.y; break;
                    }
                    acc.x += vecA.x * valB;
                    acc.y += vecA.y * valB;
                #elif WIDTH == 4
                    switch (w) {
                        case 0: valB = vecB.x; break;
                        case 1: valB = vecB.y; break;
                        case 2: valB = vecB.z; break;
                        case 3: valB = vecB.w; break;
                    }
                    acc.x += vecA.x * valB;
                    acc.y += vecA.y * valB;
                    acc.z += vecA.z * valB;
                    acc.w += vecA.w * valB;
                #elif WIDTH == 8
                    switch (w) {
                        case 0: valB = vecB.s0; break;
                        case 1: valB = vecB.s1; break;
                        case 2: valB = vecB.s2; break;
                        case 3: valB = vecB.s3; break;
                        case 4: valB = vecB.s4; break;
                        case 5: valB = vecB.s5; break;
                        case 6: valB = vecB.s6; break;
                        case 7: valB = vecB.s7; break;
                    }
                    acc.s0 += vecA.s0 * valB;
                    acc.s1 += vecA.s1 * valB;
                    acc.s2 += vecA.s2 * valB;
                    acc.s3 += vecA.s3 * valB;
                    acc.s4 += vecA.s4 * valB;
                    acc.s5 += vecA.s5 * valB;
                    acc.s6 += vecA.s6 * valB;
                    acc.s7 += vecA.s7 * valB;
                #endif
            }
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final results in C
    C[globalCol*(M/WIDTH) + globalRow] = acc;
}*/

// Simple transpose kernel for a P * Q matrix
kernel void transpose(const int P, const int Q, global const float* input,  global float* output) {
    // Thread identifiers
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    const int ID0 = get_group_id(0)*TRANSPOSEX + tx; // 0..P
    const int ID1 = get_group_id(1)*TRANSPOSEY + ty; // 0..Q

    // Set-up the local memory for shuffling
    __local float buffer[TRANSPOSEX][TRANSPOSEY];

    // Swap the x and y coordinates to perform the rotation (coalesced)
    if (ID0 < P && ID1 < Q) {
        buffer[ty][tx] = input[ID1*P + ID0];
    }

    // Synchronise all threads
    barrier(CLK_LOCAL_MEM_FENCE);

    // We don't have to swap the x and y thread indices here,
    // because that's already done in the local memory
    const int newID0 = get_group_id(1)*TRANSPOSEY + tx;
    const int newID1 = get_group_id(0)*TRANSPOSEX + ty;

    // Store the transposed result (coalesced)
    if (newID0 < Q && newID1 < P) {
        output[newID1*Q + newID0] = buffer[tx][ty];
    }
}


// Pre-transpose the input matrix B and use rectangular tiles
kernel void matrixMul5(const int M, const int N, const int K, global const float *A, global const float *B, global float *C ){
    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS/WPT == RTS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)

    // Local memory to fit a tile of A and B
    __local float Asub[TSDK][TS];
    __local float Bsub[TS][TSDK+2];

    // Initialise the accumulation registers
    float acc[WPT];
    for (int w=0; w<WPT; w++) {
        acc[w] = 0.0f;
    }
    
    // Loop over all tiles
    const int numTiles = K/TSDK;
    for (int t=0; t<numTiles; t++) {

        // Load one tile of A and B into local memory
        for (int l=0; l<LPT; l++) {
            const int tiledIndex = TSDK*t + col + l*RTS;
            int indexA = (tiledIndex)*M + TS*get_group_id(0) + row;
            int indexB = (tiledIndex)*N + TS*get_group_id(1) + row;
            Asub[col + l*RTS][row] = A[indexA];
            Bsub[row][col + l*RTS] = B[indexB];
        }

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the computation for a single tile
        for (int k=0; k<TSDK; k++) {
            for (int w=0; w<WPT; w++) {
                acc[w] += Asub[k][row] * Bsub[col + w*RTS][k];
            }
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final results in C
    for (int w=0; w<WPT; w++) {
        C[(globalCol + w*RTS)*M + globalRow] = acc[w];
    }
}


// Use 2D register blocking (further increase in work per thread)
kernel void matrixMul6(const int M, const int N, const int K, global const float *A, global const float *B, global float *C ){
    // Thread identifiers
    const int tidm = get_local_id(0); // Local row ID (max: TSM/WPTM == RTSM)
    const int tidn = get_local_id(1); // Local col ID (max: TSN/WPTN == RTSN)
    const int offsetM = TSM*get_group_id(0); // Work-group offset
    const int offsetN = TSN*get_group_id(1); // Work-group offset

    // Local memory to fit a tile of A and B
    __local float Asub[TSK][TSM];
    __local float Bsub[TSN][TSK+2];

    // Allocate register space
    float Areg;
    float Breg[WPTN];
    float acc[WPTM][WPTN];

    // Initialise the accumulation registers
    #pragma unroll
    for (int wm=0; wm<WPTM; wm++) {
        #pragma unroll
        for (int wn=0; wn<WPTN; wn++) {
            acc[wm][wn] = 0.0f;
        }
    }
    
    // Loop over all tiles
    const int numTiles = K/TSK;
    int t=0;
    do {

        // Load one tile of A and B into local memory
        #pragma unroll
        for (int la=0; la<LPTA; la++) {
            int tid = tidn*RTSM + tidm;
            int id = la*RTSN*RTSM + tid;
            int row = MOD2(id,TSM);
            int col = DIV2(id,TSM);
            int tiledIndex = TSK*t + col;
            Asub[col][row] = A[tiledIndex*M + offsetM + row];
            Bsub[row][col] = B[tiledIndex*N + offsetN + row];
        }

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Loop over the values of a single tile
        for (int k=0; k<TSK; k++) {

            // Cache the values of Bsub in registers
            #pragma unroll
            for (int wn=0; wn<WPTN; wn++) {
                int col = tidn + wn*RTSN;
                Breg[wn] = Bsub[col][k];
            }

            // Perform the computation
            #pragma unroll
            for (int wm=0; wm<WPTM; wm++) {
                int row = tidm + wm*RTSM;
                Areg = Asub[k][row];
                #pragma unroll
                for (int wn=0; wn<WPTN; wn++) {
                    acc[wm][wn] += Areg * Breg[wn];
                }
            }
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);

        // Next tile
        t++;
    } while (t<numTiles);

    // Store the final results in C
    #pragma unroll
    for (int wm=0; wm<WPTM; wm++) {
        int globalRow = offsetM + tidm + wm*RTSM;
        #pragma unroll
        for (int wn=0; wn<WPTN; wn++) {
            int globalCol = offsetN + tidn + wn*RTSN;
            C[globalCol*M + globalRow] = acc[wm][wn];
        }
    }
}


