#include "clFramework/clApp.hpp"
#include <iomanip>

//#define DIM 2048
#define DIM 4096
//#define DIM 32768 //2(32768 = 1<<15)

void CPUSingleThreadMatVecMul(int M, int N, std::vector<float> &matrixA, std::vector<float> &vectorB, std::vector<float> &outputVector, int sampleNum){
    int count = 0;
	int printDelta = sampleNum / 5;
	
	for(int m = 0; m < M; m++){
		float acc = 0.0f;
    	for (int n = 0; n < N; n++) 
        	acc += matrixA[m*M + n] * vectorB[n];
		outputVector[m] = acc;

		count++;
		if(count % printDelta == 0){
			float completeRate = (count * 100.0)/sampleNum ;
			std::cout<<"Completed: "<<completeRate<<"%"<<std::endl;
		}
			
        if(count >= sampleNum) return;
	}
}

int main() {
	CTimer timer;
	timer.initialize();
	
	srand(time(NULL));

	CCLAPP clApp(false, true, true);//verbose, profiler, verify
	clApp.initDevice();
	clApp.loadShader("matrixVectorMul.cl");// matrixA(m by n) * vectorB(n by 1) = vectorC(m by 1)
	clApp.buildProgram();

	//Step 1: Create kernel program from shader function
	cl::Kernel program_kernel(clApp.program, "matrixVectorMul");

	if(clApp.bProfiler) timer.printDeltaTime("Initializazion done");

	//Step 2: Allocate host buffers, and fill with random numbers
	const int matrixDimM = DIM; 
	const int matrixDimN = DIM;
	std::vector<float> a_host(matrixDimM*matrixDimN, 1); 
	std::vector<float> b_host(matrixDimN, 1); 
	std::vector<float> c_host(matrixDimM); 


	for (int i=0; i<matrixDimM*matrixDimN; i++) 
		a_host[i] = (float)rand() / (float)RAND_MAX;
	for (int i=0; i<matrixDimN; i++) 
		b_host[i] = (float)rand() / (float)RAND_MAX;
	

	if(clApp.bVerbose) PrintMatrix("Matrix A: ", a_host, matrixDimM, matrixDimN);
	if(clApp.bVerbose) PrintVector("Vector B: ", b_host, matrixDimN);

	if(clApp.bProfiler) timer.printDeltaTime("Allocate host buffer done");

	//Step 3: host >> device (Allocate device buffers and transfer data) 
	cl::Buffer A_device(clApp.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		a_host.size() * sizeof(float), a_host.data());
	cl::Buffer B_device(clApp.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		b_host.size() * sizeof(float), b_host.data());
	cl::Buffer C_device(clApp.context, CL_MEM_READ_WRITE,
		c_host.size() * sizeof(float));

	if(clApp.bProfiler) timer.printDeltaTime("Host >> Device");

	//Step 4: Set kernel parameters.
	program_kernel.setArg(0, matrixDimM);
	program_kernel.setArg(1, matrixDimN);
	program_kernel.setArg(2, A_device);
	program_kernel.setArg(3, B_device);
	program_kernel.setArg(4, C_device);
	
	//Step 5: Launch kernel on the compute device.
	cl::NDRange global(matrixDimM, matrixDimN);
	clApp.queue.enqueueNDRangeKernel(program_kernel, cl::NullRange, global, cl::NullRange);
	clApp.queue.finish();//block host until device finishes

	if(clApp.bProfiler) timer.printDeltaTime("Kernel run done");

	//Step 6: device >> host
	clApp.queue.enqueueReadBuffer(C_device, CL_TRUE, 0, c_host.size() * sizeof(float), c_host.data());

	if(clApp.bProfiler) timer.printDeltaTime("Device >> Host");

	if(clApp.bVerbose) PrintVector("Vector C: ", c_host, matrixDimM);

	//Verify Correctness
	if(clApp.bVerify){
		int sampleNum = 100 > matrixDimM ? matrixDimM : 100;
		std::cout<<"sampleNum: "<<sampleNum<<std::endl;
		float threshold = 0.0001f;
		std::cout<<"threshold: "<<threshold<<std::endl;

		std::vector<float> outputVector(matrixDimM); 
        CPUSingleThreadMatVecMul(matrixDimM, matrixDimN, a_host, b_host, outputVector, sampleNum);
		if(clApp.bProfiler) timer.printDeltaTime("---Profiler: CPU single thread calculation done");

		std::cout<<"Verification begin."<<std::endl;
		for (int i=0; i<sampleNum; i++) {
			//std::cout<<c_host[i]<<std::endl;
			//std::cout<<outputVector[i]<<std::endl;
			float diff = std::abs(outputVector[i]-c_host[i]);
			if(diff > threshold)
				std::cout<<"i="<<i<<std::setprecision(10) <<", Host: "<<outputVector[i]<<", Device: "<<c_host[i]<<", Diff: "<<diff<<std::endl;
		}
		std::cout<<"Verification done."<<std::endl;
	}


	return 1;
}
