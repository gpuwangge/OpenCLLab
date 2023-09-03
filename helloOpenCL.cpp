#include "clFramework\\clApp.hpp"


int main() {
	CCLAPP clApp;
	clApp.init();

	// Compute c = a + b.
    try {
		cl::Kernel add(clApp.program, "add");

		// Prepare input data.
		std::vector<float> a_host(clApp.NDRange, 1); //double
		std::vector<float> b_host(clApp.NDRange, 2); //double
		std::vector<float> c_host(clApp.NDRange); //double

		// Allocate device buffers and transfer data host >> device.
		cl::Buffer A_device(clApp.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			a_host.size() * sizeof(float), a_host.data());

		cl::Buffer B_device(clApp.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			b_host.size() * sizeof(float), b_host.data());

		cl::Buffer C_device(clApp.context, CL_MEM_READ_WRITE,
			c_host.size() * sizeof(float));

		// Set kernel parameters.
		add.setArg(0, static_cast<cl_ulong>(clApp.NDRange));
		add.setArg(1, A_device);
		add.setArg(2, B_device);
		add.setArg(3, C_device);
		
		// Launch kernel on the compute device.
		clApp.queue.enqueueNDRangeKernel(add, cl::NullRange, clApp.NDRange, cl::NullRange);

		// Get result device >> host.
		clApp.queue.enqueueReadBuffer(C_device, CL_TRUE, 0, c_host.size() * sizeof(float), c_host.data());

		// Should get '3' here.
		std::cout << c_host[134224] << std::endl;
    } catch (const cl::Error &err) {
		std::cerr
			<< "OpenCL error: "
			<< err.what() << "(" << err.err() << ")"
			<< std::endl;
		return 0;
    }

	return 1;
}
