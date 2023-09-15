#include "clFramework/clApp.hpp"

int main() {
	CCLAPP clApp(true, true, true);
	clApp.initDevice();
	clApp.loadShader("vectorAdd.cl");// Compute c = a + b.
	clApp.buildProgram();

	
    //try {

	//Step 1: Create kernel program from shader function
	cl::Kernel program_kernel(clApp.program, "vectorAdd");

	//Step 2: Allocate host buffers
	std::vector<float> a_host(clApp.maxNDRange, 1); //double
	std::vector<float> b_host(clApp.maxNDRange, 2); //double
	std::vector<float> c_host(clApp.maxNDRange); //double

	//Step 3: host >> device (Allocate device buffers and transfer data) 
	cl::Buffer A_device(clApp.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		a_host.size() * sizeof(float), a_host.data());
	cl::Buffer B_device(clApp.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		b_host.size() * sizeof(float), b_host.data());
	cl::Buffer C_device(clApp.context, CL_MEM_READ_WRITE,
		c_host.size() * sizeof(float));

	//Step 4: Set kernel parameters.
	program_kernel.setArg(0, static_cast<cl_ulong>(clApp.maxNDRange));
	program_kernel.setArg(1, A_device);
	program_kernel.setArg(2, B_device);
	program_kernel.setArg(3, C_device);
	
	//Step 5: Launch kernel on the compute device.
	clApp.queue.enqueueNDRangeKernel(program_kernel, cl::NullRange, clApp.maxNDRange, cl::NullRange);
	clApp.queue.finish();//block host until device finishes

	//Step 6: device >> host
	clApp.queue.enqueueReadBuffer(C_device, CL_TRUE, 0, c_host.size() * sizeof(float), c_host.data());

	// Should get '3' here.
	std::cout << "Result is: " << c_host[134224] << std::endl;

    //} catch (const cl::Error &err) {
	//	std::cerr
	//		<< "OpenCL error: "
	//		<< err.what() << "(" << err.err() << ")"
	//		<< std::endl;
	//	return 0;
    //}

	return 1;
}
