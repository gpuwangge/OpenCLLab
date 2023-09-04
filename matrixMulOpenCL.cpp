#include "clFramework\\clApp.hpp"

//#define DIM 4096
//#define DIM 8192
//#define DIM 16384
#define DIM 32768


void CPUSingleThreadMatMul(int M, int N, int K, std::vector<float> &matrixA, std::vector<float> &matrixB, std::vector<float> &outputMatrix, int sampleNum){
    int count = 0;
    for(int m = 0; m < M; m++){ //row
        for(int n = 0; n < N; n++){ //col
            //row major matrix multiplication
            /*
            outputMatrix[m*N + n] = 0;
            for(int k = 0; k < K; k++){
                outputMatrix[m*N + n] += matrixA[m*K + k] * matrixB[k*N + n];
            }
            */
            //column major matrix multiplication
            outputMatrix[n*M + m] = 0;
            for(int k = 0; k < K; k++){
                outputMatrix[n*M + m] += matrixA[k*M + m] * matrixB[n*K + k];
            }
            count++;
            if(count >= sampleNum) return;
        }
    }
}

int main() {
	CTimer timer;
	timer.initialize();
	
	srand(time(NULL));

	CCLAPP clApp(false, true, false);//verbose, profiler, verify
	clApp.initDevice();
	clApp.loadShader("matrixMul.cl");// Compute c = a + b.
	clApp.buildProgram();

	//Step 1: Create kernel program from shader function
	cl::Kernel program_kernel(clApp.program, "matrixMul");

	if(clApp.bProfiler) timer.printDeltaTime("Initializazion done");

	//Step 2: Allocate host buffers, and fill with random numbers
	const int matrixDimM = DIM; 
	const int matrixDimK = DIM;
	const int matrixDimN = DIM;
	std::vector<float> a_host(matrixDimM*matrixDimK); 
	std::vector<float> b_host(matrixDimK*matrixDimN); 
	std::vector<float> c_host(matrixDimM*matrixDimN); 

	for (int i=0; i<matrixDimM*matrixDimK; i++) {
		a_host[i] = (float)rand() / (float)RAND_MAX;
	}
	for (int i=0; i<matrixDimK*matrixDimN; i++) {
		b_host[i] = (float)rand() / (float)RAND_MAX;
	}

	if(clApp.bVerbose) PrintMatrix("Matrix A: ", a_host, matrixDimM, matrixDimK);
	if(clApp.bVerbose) PrintMatrix("Matrix B: ", b_host, matrixDimK, matrixDimN);

	if(clApp.bProfiler) timer.printDeltaTime("Allocate host buffer done");

	//Step 3: host >> device (Allocate device buffers and transfer data) 
	cl::Buffer A_device(clApp.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		a_host.size() * sizeof(float), a_host.data());
	cl::Buffer B_device(clApp.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		b_host.size() * sizeof(float), b_host.data());
	cl::Buffer C_device(clApp.context, CL_MEM_READ_WRITE,
		c_host.size() * sizeof(float));

	if(clApp.bProfiler) timer.printDeltaTime("Transfer data to device done");

	//Step 4: Set kernel parameters.
	program_kernel.setArg(0, matrixDimM);
	program_kernel.setArg(1, matrixDimN);
    program_kernel.setArg(2, matrixDimK);
	program_kernel.setArg(3, A_device);
	program_kernel.setArg(4, B_device);
	program_kernel.setArg(5, C_device);
	
	//Step 5: Launch kernel on the compute device.
    cl::NDRange global(matrixDimM, matrixDimN);
	clApp.queue.enqueueNDRangeKernel(program_kernel, cl::NullRange, global, cl::NullRange);
	clApp.queue.finish();//block host until device finishes

	if(clApp.bProfiler) timer.printDeltaTime("Kernel run done");

	//Step 6: device >> host
	clApp.queue.enqueueReadBuffer(C_device, CL_TRUE, 0, c_host.size() * sizeof(float), c_host.data());

	if(clApp.bProfiler) timer.printDeltaTime("Transfer data back to host done");

	if(clApp.bVerbose) PrintMatrix("Matrix C: ", c_host, matrixDimM, matrixDimN);

	//Verify Correctness
	if(clApp.bVerify){
        std::cout<<"Verify Correctness"<<std::endl;
		int sampleNum = 100 > matrixDimM*matrixDimN ? matrixDimM*matrixDimN : 100;
        std::vector<float> outputMatrix(matrixDimM*matrixDimN); 
        CPUSingleThreadMatMul(matrixDimM, matrixDimN, matrixDimK, a_host, b_host, outputMatrix, sampleNum);

		for (int i=0; i<sampleNum; i++) {
			float diff = outputMatrix[i]-c_host[i];
			float threshold = 0.00001f;
			if(diff > threshold)
				std::cout<<"Host: "<<outputMatrix[i]<<", Device: "<<c_host[i]<<", Diff: "<<diff<<std::endl;
		}
	}


	return 1;
}
