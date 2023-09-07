#include "clFramework\\clApp.hpp"
#include <iomanip>

#define DIM 4096
//#define DIM 8192
//#define DIM 16384
//#define DIM 32768

#define TILESIZE 32  //Tile Size: for 1080 TI, CL_DEVICE_MAX_WORK_GROUP_SIZE=1024=32x32, so 32 is maximum TS value.
//#define TILESIZE 16   //for Iris GPU

enum KernelModes 
{   KERNEL1 = 0, //Naive implementation
    KERNEL2 = 1, //Tiling in the local memory
    KERNEL3 = 2, //More work per thread
    KERNEL4 = 3, //Wider data-types
	KERNEL5 = 4, //Transposed input matrix and rectangular tiles
    KERNEL6 = 5	 //2D register blocking
};

// Constants for kernels 4, 7 -- 10
#define WIDTH 4                      // The vector-width (in number of floats)

// Constants for the supporting transpose kernel
#define TRANSPOSEX 16
#define TRANSPOSEY 16

// Constants for kernels 6 -- 10
#define TSM 128                      // The tile-size in dimension M
#define TSN 128                      // The tile-size in dimension N
#define TSK 16                       // The tile-size in dimension K
#define WPTM 8                       // The amount of work-per-thread in dimension M
#define WPTN 8                       // The amount of work-per-thread in dimension N

void CPUSingleThreadMatMul(int M, int N, int K, std::vector<float> &matrixA, std::vector<float> &matrixB, std::vector<float> &outputMatrix, int sampleNum){
    int count = 0;
	int printDelta = sampleNum / 5;
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
			if(count % printDelta == 0){
				float completeRate = (count * 100.0)/sampleNum ;
				std::cout<<"Completed: "<<completeRate<<"%"<<std::endl;
			}
			
            if(count >= sampleNum) return;
        }
    }
}

int main() {
	CTimer timer;
	timer.initialize();
	
	srand(time(NULL));

	CCLAPP clApp(false, true, true);//verbose, profiler, verify
	clApp.initDevice();
	clApp.loadShader("matrixMul.cl");// Compute c = a*b.
	clApp.buildProgram();

	KernelModes kernelMode = KERNEL4;

	//Step 1: Create kernel program from shader function
	cl::Kernel program_kernel;
	std::string kernelName = "matrixMul" + std::to_string(kernelMode+1);
	program_kernel = cl::Kernel(clApp.program, kernelName.c_str());

	cl::Kernel program_transpose;
	if(kernelMode == KERNEL5|KERNEL6)
		program_transpose = cl::Kernel(clApp.program, "transpose");
	
	if(clApp.bProfiler) timer.printDeltaTime("---Profiler: Initializazion done");

	//Step 2: Allocate host buffers, and fill with random numbers
	const int matrixDimM = DIM; 
	const int matrixDimK = DIM;
	const int matrixDimN = DIM;
	std::vector<float> a_host(matrixDimM*matrixDimK); 
	std::vector<float> b_host(matrixDimK*matrixDimN); 
	std::vector<float> c_host(matrixDimM*matrixDimN); 

	for (int i=0; i<matrixDimM*matrixDimK; i++) 
		a_host[i] = (float)rand() / (float)RAND_MAX;
	for (int i=0; i<matrixDimK*matrixDimN; i++) 
		b_host[i] = (float)rand() / (float)RAND_MAX;

	if(clApp.bVerbose) PrintMatrix("Matrix A: ", a_host, matrixDimM, matrixDimK);
	if(clApp.bVerbose) PrintMatrix("Matrix B: ", b_host, matrixDimK, matrixDimN);

	if(clApp.bProfiler) timer.printDeltaTime("---Profiler: Allocate host buffer done");

	//Step 3: host >> device (Allocate device buffers and transfer data) 
	cl::Buffer A_device(clApp.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		a_host.size() * sizeof(float), a_host.data());
	cl::Buffer B_device(clApp.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		b_host.size() * sizeof(float), b_host.data());
	cl::Buffer C_device(clApp.context, CL_MEM_READ_WRITE, //?why write?: copy value to erase the previous run if needed 
		c_host.size() * sizeof(float));

	//Transpose B for Kernel5&6
	cl::Buffer B_TR_device(clApp.context, CL_MEM_READ_WRITE, //?change to read only?
		b_host.size() * sizeof(float));

	if(clApp.bProfiler) timer.printDeltaTime("---Profiler: Host >> Device");

	//Step 4: Set kernel parameters.
	if(kernelMode == KERNEL5|KERNEL6){
		program_transpose.setArg(0, matrixDimK);
		program_transpose.setArg(1, matrixDimN);
		program_transpose.setArg(2, B_device);
		program_transpose.setArg(3, B_TR_device);		
	}

	program_kernel.setArg(0, matrixDimM);
	program_kernel.setArg(1, matrixDimN);
    program_kernel.setArg(2, matrixDimK);
	program_kernel.setArg(3, A_device);
	if(kernelMode == KERNEL5|KERNEL6) program_kernel.setArg(4, B_TR_device);
	else program_kernel.setArg(4, B_device);
	program_kernel.setArg(5, C_device);

	if(clApp.bProfiler) timer.printDeltaTime("---Profiler: Set kernel parameters");
	
	//Step 5: Launch kernel on the compute device.
	const int WPT = 8; //for kernel 3
	cl::NDRange local(TILESIZE, TILESIZE);
    cl::NDRange global(matrixDimM, matrixDimN);

	switch(kernelMode){
	case KERNEL1:
		break;
	case KERNEL2:
		break;
	case KERNEL3:
		local = cl::NDRange(TILESIZE, TILESIZE/WPT);
    	global = cl::NDRange(matrixDimM, matrixDimN/WPT);
		break;
	case KERNEL4:
		local = cl::NDRange(TILESIZE/WIDTH, TILESIZE);
    	global = cl::NDRange((size_t)(matrixDimM/WIDTH), (size_t)matrixDimN);
		break;
	case KERNEL5:
		local = cl::NDRange(TILESIZE, TILESIZE/WPT);
    	global = cl::NDRange(matrixDimM, matrixDimN/WPT);
		break;
	case KERNEL6:
		local = cl::NDRange(TSM/WPTM, TSN/WPTN);
    	global = cl::NDRange(matrixDimM/WPTM, matrixDimN/WPTN);
		break;
	default:
		break;
	}

	if(kernelMode == KERNEL5|KERNEL6){
		cl::NDRange transposeLocal(TRANSPOSEX, TRANSPOSEY);
    	cl::NDRange transposeGlobal(matrixDimK, matrixDimN);
		clApp.queue.enqueueNDRangeKernel(program_transpose, cl::NullRange, transposeGlobal, transposeLocal);
	}

	clApp.queue.enqueueNDRangeKernel(program_kernel, cl::NullRange, global, local);
	clApp.queue.finish();//block host until device finishes

	if(clApp.bProfiler) timer.printDeltaTime("---Profiler: Kernel run done");

	//Step 6: device >> host
	clApp.queue.enqueueReadBuffer(C_device, CL_TRUE, 0, c_host.size() * sizeof(float), c_host.data());

	if(clApp.bProfiler) timer.printDeltaTime("---Profiler: Device >> Host");

	if(clApp.bVerbose) PrintMatrix("Matrix C: ", c_host, matrixDimM, matrixDimN);

	//Verify Correctness
	if(clApp.bVerify){
		int sampleNum = 10000 > matrixDimM*matrixDimN ? matrixDimM*matrixDimN : 10000;
		//int sampleNum = matrixDimM*matrixDimN;
		std::cout<<"sampleNum: "<<sampleNum<<std::endl;
		float threshold = 0.000001f;
		std::cout<<"threshold: "<<threshold<<std::endl;

        std::vector<float> outputMatrix(matrixDimM*matrixDimN); 
        CPUSingleThreadMatMul(matrixDimM, matrixDimN, matrixDimK, a_host, b_host, outputMatrix, sampleNum);
		if(clApp.bProfiler) timer.printDeltaTime("---Profiler: CPU single thread calculation done");

		std::cout<<"Verification begin."<<std::endl;
		for (int i=0; i<sampleNum; i++) {
			float diff = outputMatrix[i]-c_host[i];
			if(diff > threshold)
				std::cout<<"i="<<i<<std::setprecision(10) <<", Host: "<<outputMatrix[i]<<", Device: "<<c_host[i]<<", Diff: "<<diff<<std::endl;
		}
		std::cout<<"Verification done."<<std::endl;
	}


	return 1;
}
