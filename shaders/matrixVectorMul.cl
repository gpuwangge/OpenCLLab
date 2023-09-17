kernel void matrixVectorMul(const int M, const int N, global const float *A, global const float *B, global float *C ){
    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)

    // Compute a single element (loop over K)
    // matrixA(m by n) * vectorB(n by 1) = vectorC(m by 1)
    float acc = 0.0f;
    for (int i=0; i<N; i++) 
        acc += A[globalRow*M + i] * B[i];
    
    // Store the result
    C[globalRow] = acc;
}