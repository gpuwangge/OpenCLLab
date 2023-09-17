#ifndef H_UTILITY
#define H_UTILITY

#include <CL/opencl.hpp>
#include <iostream>
#include <chrono>

static cl_int PrintPlatformInfoSummary(cl::Platform platform)
{
    std::cout << "\tName:           " << platform.getInfo<CL_PLATFORM_NAME>()
              << "\n";
    std::cout << "\tVendor:         " << platform.getInfo<CL_PLATFORM_VENDOR>()
              << "\n";
    std::cout << "\tDriver Version: " << platform.getInfo<CL_PLATFORM_VERSION>()
              << "\n";

    return CL_SUCCESS;
}

static void PrintDeviceType(const std::string& label, cl_device_type type)
{
    std::cout << label << ((type & CL_DEVICE_TYPE_DEFAULT) ? "DEFAULT " : "")
              << ((type & CL_DEVICE_TYPE_CPU) ? "CPU " : "")
              << ((type & CL_DEVICE_TYPE_GPU) ? "GPU " : "")
              << ((type & CL_DEVICE_TYPE_ACCELERATOR) ? "ACCELERATOR " : "")
              << ((type & CL_DEVICE_TYPE_CUSTOM) ? "CUSTOM " : "") << "\n";
}

static cl_int PrintDeviceInfoSummary(const std::vector<cl::Device> devices)
{
    for (size_t i = 0; i < devices.size(); i++)
    {
        std::cout << "Device[" << i << "]:\n";

        cl_device_type deviceType = devices[i].getInfo<CL_DEVICE_TYPE>();
        PrintDeviceType("\tType:           ", deviceType);

        std::cout << "\tName:           "
                  << devices[i].getInfo<CL_DEVICE_NAME>() << "\n";
        std::cout << "\tVendor:         "
                  << devices[i].getInfo<CL_DEVICE_VENDOR>() << "\n";
        std::cout << "\tDevice Version: "
                  << devices[i].getInfo<CL_DEVICE_VERSION>() << "\n";
        std::cout << "\tDevice Profile: "
                  << devices[i].getInfo<CL_DEVICE_PROFILE>() << "\n";
        std::cout << "\tDriver Version: "
                  << devices[i].getInfo<CL_DRIVER_VERSION>() << "\n";
        std::cout << "\tCL_DEVICE_MAX_WORK_GROUP_SIZE: "
                  << devices[i].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << "\n";
        std::cout << "\tCL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: "
                  << devices[i].getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>() << "\n";          
        std::cout << "\tCL_DEVICE_MAX_WORK_ITEM_SIZES[0]: "
                  << devices[i].getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[0] << "\n";
        std::cout << "\tCL_DEVICE_MAX_WORK_ITEM_SIZES[1]: "
                  << devices[i].getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[1] << "\n";
        std::cout << "\tCL_DEVICE_MAX_WORK_ITEM_SIZES[2]: "
                  << devices[i].getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[2] << "\n";
        std::cout << "\tCL_DEVICE_MAX_COMPUTE_UNITS: "
                  << devices[i].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << "\n";
    }

    return CL_SUCCESS;
}

static void PrintMatrix(std::string message, std::vector<float> &matrix, int M, int N){
    std::cout<<message<<std::endl;
	//Row id is M; Col id is N
	for(int i = 0; i < M; i++){
		for(int j = 0; j < N; j++){
			int index = i * M + j;
			std::cout<<matrix[index]<<" ";
		}
		std::cout<<std::endl;
	}
}

static void PrintVector(std::string message, std::vector<float> &vector, int N){
    std::cout<<message<<std::endl;
	for(int i = 0; i < N; i++)
		std::cout<<vector[i]<<" ";
    std::cout<<std::endl;
}

class CTimer final{
public:
	CTimer(){}
	~CTimer(){}

	decltype(std::chrono::high_resolution_clock::now()) startTime;
	decltype(std::chrono::high_resolution_clock::now()) lastTime;

	void initialize(){
		startTime = std::chrono::high_resolution_clock::now();
		lastTime = std::chrono::high_resolution_clock::now();
	}

	void printDeltaTime(std::string message){
		auto currentTime = std::chrono::high_resolution_clock::now();
		auto deltaTime = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - lastTime).count();
		lastTime = currentTime;
		std::cout<< message << ", time elapsed: "<<deltaTime<< "s"<<std::endl;
	}
};

#endif