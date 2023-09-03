#ifndef H_UTILITY
#define H_UTILITY

#include <CL/opencl.hpp>
#include <iostream>

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
    }

    return CL_SUCCESS;
}

static void PrintMatrix(std::vector<float> &matrix, int M, int N){
	//Row id is M; Col id is N
	for(int i = 0; i < M; i++){
		for(int j = 0; j < N; j++){
			int index = i * M + j;
			std::cout<<matrix[index]<<" ";
		}
		std::cout<<std::endl;
	}
}

#endif