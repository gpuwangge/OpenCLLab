#include "clFramework/clApp.hpp"
#include <iomanip>

int main() {
	CTimer timer;
	timer.initialize();
	
	srand(time(NULL));

	CCLAPP clApp(true, true, false);//verbose, profiler, verify
	clApp.initDevice();
	clApp.loadShader("imageIO.cl");
	clApp.buildProgram();

	//Step 1: Create kernel program from shader function
	cl::Kernel program_kernel;
	std::string kernelName = "read_write_image";
	program_kernel = cl::Kernel(clApp.program, kernelName.c_str());
	
	if(clApp.bProfiler) timer.printDeltaTime("---Profiler: Initializazion done");

	//Step 2: Allocate host buffers, and fill with random numbers
	if(clApp.bVerbose) std::cout<<"imageA_host: "<<std::endl;
	std::vector<float> imageA_host(256, 0.0);
	for(int i = 0; i < 256; i++){
		imageA_host[i] = 255.0 - i;
		if(clApp.bVerbose) std::cout<<imageA_host[i]<<" ";
	}
	if(clApp.bVerbose) std::cout<<std::endl;

	std::vector<float> imageB_host(256, 0.0);

	if(clApp.bProfiler) timer.printDeltaTime("---Profiler: Allocate host buffer done");

	//Step 3: host >> device (Allocate device buffers and transfer data) 
	cl::ImageFormat grayscale(CL_R, CL_FLOAT);
	cl::Image2D Input_Image(clApp.context, CL_MEM_READ_ONLY, grayscale, 16, 16);
	cl::Image2D Output_Image(clApp.context, CL_MEM_WRITE_ONLY, grayscale, 16, 16);
	std::array<cl::size_type, 3> origin {0,0,0};
	std::array<cl::size_type, 3> region {16,16,1};
	clApp.queue.enqueueWriteImage(Input_Image, CL_TRUE, origin, region, 0, 0, &imageA_host[0]);

	if(clApp.bProfiler) timer.printDeltaTime("---Profiler: Host >> Device");

	//Step 4: Set kernel parameters.
	program_kernel.setArg(0, Input_Image);
	program_kernel.setArg(1, Output_Image);

	//if(clApp.bProfiler) timer.printDeltaTime("---Profiler: Set kernel parameters");
	
	//Step 5: Launch kernel on the compute device.
	clApp.queue.enqueueNDRangeKernel(program_kernel, cl::NullRange, cl::NDRange(16,16), cl::NullRange, NULL);
	clApp.queue.finish();//block host until device finishes

	if(clApp.bProfiler) timer.printDeltaTime("---Profiler: Kernel run done");

	//Step 6: device >> host
	clApp.queue.enqueueReadImage(Output_Image, CL_TRUE, origin, region, 0, 0, &imageB_host[0]);

	if(clApp.bProfiler) timer.printDeltaTime("---Profiler: Device >> Host");

	//Verify Correctness
	if(clApp.bVerbose) std::cout<<"iamgeB_host: "<<std::endl;
	for(int i = 0; i < 256; i++)
		if(clApp.bVerbose) std::cout<<imageB_host[i]<<" ";
	if(clApp.bVerbose) std::cout<<std::endl;

	std::cout<<"Done. "<<std::endl;
	return 0;
}
