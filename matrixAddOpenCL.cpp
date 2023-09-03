#include "clFramework\\clApp.hpp"

int main() {
	srand(time(NULL));

	CCLAPP clApp;
	clApp.initDevice();
	clApp.loadShader("matrixAdd.cl");// Compute c = a + b.
	clApp.buildProgram();

	//Step 1: Create kernel program from shader function
	cl::Kernel program_kernel(clApp.program, "matrixAdd");

	//Step 2: Allocate host buffers, and fill with random numbers
	const int matrixDimM = 5; //mxk squares + kxn squares
	const int matrixDimK = 5;
	const int matrixDimN = 5;
	std::vector<float> a_host(matrixDimM*matrixDimK); 
	std::vector<float> b_host(matrixDimK*matrixDimN); 
	std::vector<float> c_host(matrixDimM*matrixDimN); 

	for (int i=0; i<matrixDimM*matrixDimK; i++) {
		a_host[i] = (float)rand() / (float)RAND_MAX;
	}
	for (int i=0; i<matrixDimK*matrixDimN; i++) {
		b_host[i] = (float)rand() / (float)RAND_MAX;
	}

	std::cout<<"Matrix A: "<<std::endl;
	PrintMatrix(a_host, matrixDimM, matrixDimK);
	std::cout<<"Matrix B: "<<std::endl;
	PrintMatrix(b_host, matrixDimK, matrixDimN);

	//Step 3: host >> device (Allocate device buffers and transfer data) 
	cl::Buffer A_device(clApp.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		a_host.size() * sizeof(float), a_host.data());
	cl::Buffer B_device(clApp.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		b_host.size() * sizeof(float), b_host.data());
	cl::Buffer C_device(clApp.context, CL_MEM_READ_WRITE,
		c_host.size() * sizeof(float));

	//Step 4: Set kernel parameters.
	program_kernel.setArg(0, matrixDimM);
	program_kernel.setArg(1, matrixDimN);
	program_kernel.setArg(2, A_device);
	program_kernel.setArg(3, B_device);
	program_kernel.setArg(4, C_device);
	
	//Step 5: Launch kernel on the compute device.
	clApp.queue.enqueueNDRangeKernel(program_kernel, cl::NullRange, matrixDimM*matrixDimN, cl::NullRange);

	//Step 6: device >> host
	clApp.queue.enqueueReadBuffer(C_device, CL_TRUE, 0, c_host.size() * sizeof(float), c_host.data());

	// Should get '3' here.
	std::cout<<"Matrix C: "<<std::endl;
	PrintMatrix(c_host, matrixDimM, matrixDimN);

	return 1;
}
