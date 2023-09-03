#include "clFramework\\clApp.hpp"

//#define DIM 32768 //mxk squares + kxn squares, use power of 2(32768 = 1<<15)
//NVIDIA GTX 1080 TI: allocate host buffer: 28s, host>>device: 1.4s, kernel: 2.5s, device>>host: 1.4s
//Intel Xe Graphics: allocate host buffer: 47s, host >> device error: clCreateBuffer() throw an insance error

#define DIM 16384
//NVIDIA GTX 1080 TI: allocate host buffer: 7s, host>>device: 0.34s, kernel: 0.55s, device>>host: 0.21s
//Intel Xe Graphics: allocate host buffer: 9s, host>>device: 0.7s, kernel: 0.001s(???), device>>host: 0.6s

int main() {
	CTimer timer;
	timer.initialize();
	
	srand(time(NULL));

	CCLAPP clApp(false, true, true);
	clApp.initDevice();
	clApp.loadShader("matrixAdd.cl");// Compute c = a + b.
	clApp.buildProgram();

	//Step 1: Create kernel program from shader function
	cl::Kernel program_kernel(clApp.program, "matrixAdd");

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
	program_kernel.setArg(2, A_device);
	program_kernel.setArg(3, B_device);
	program_kernel.setArg(4, C_device);
	
	//Step 5: Launch kernel on the compute device.
	clApp.queue.enqueueNDRangeKernel(program_kernel, cl::NullRange, matrixDimM*matrixDimN, cl::NullRange);

	if(clApp.bProfiler) timer.printDeltaTime("Kernel run done");

	//Step 6: device >> host
	clApp.queue.enqueueReadBuffer(C_device, CL_TRUE, 0, c_host.size() * sizeof(float), c_host.data());

	if(clApp.bProfiler) timer.printDeltaTime("Transfer data back to host done");

	if(clApp.bVerbose) PrintMatrix("Matrix C: ", c_host, matrixDimM, matrixDimN);

	//Verify Correctness
	if(clApp.bVerify){
		int sampleNum = 100;
		for (int i=0; i<sampleNum; i++) {
			float real = a_host[i]+b_host[i];
			float diff = real-c_host[i];
			float threshold = 0.0f;
			if(diff > threshold)
				std::cout<<"Host: "<<real<<", Device: "<<c_host[i]<<", Diff: "<<diff<<std::endl;
		}
	}


	return 1;
}
