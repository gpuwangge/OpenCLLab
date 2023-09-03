kernel void matrixAdd(
        const int M, const int N, 
        global const float *A, 
        global const float *B, 
        global float *C 
        )
{
    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)

    size_t i = globalCol*M + globalRow;
    C[i] = A[i] + B[i];
}