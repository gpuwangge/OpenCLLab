#ifndef H_CLAPP
#define H_CLAPP

#include <iostream>
#include <vector>
#include <string>

//#define __CL_ENABLE_EXCEPTIONS
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>

//#include <CL/Utils/Utils.hpp>
#include<fstream>

#define SHADER_PATH "../shaders/"

/*
static const char shaderCode[] =
    "#if defined(cl_khr_fp64)\n"
    "#  pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
    "#elif defined(cl_amd_fp64)\n"
    "#  pragma OPENCL EXTENSION cl_amd_fp64: enable\n"
    "#else\n"
    "#  error double precision is not supported\n"
    "#endif\n"
    "kernel void add(\n"
    "       ulong n,\n"
    "       global const double *a,\n"
    "       global const double *b,\n"
    "       global double *c\n"
    "       )\n"
    "{\n"
    "    size_t i = get_global_id(0);\n"
    "    if (i < n) {\n"
    "       c[i] = a[i] + b[i];\n"
    "    }\n"
    "}\n";*/

class CCLAPP{
public:
    CCLAPP();
    ~CCLAPP();

    bool init();

    bool bVerbose;
    const size_t NDRange= 1 << 20;

	cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;

	bool readFile(const std::string& filename, std::string &buffer);

private:
	std::vector<cl::Platform> platform;
    std::vector<cl::Device> devices;	
};

CCLAPP::CCLAPP(){}
CCLAPP::~CCLAPP(){}

bool CCLAPP::init(){
    bVerbose = false;
	if(bVerbose) std::cout<<"NDRange: "<<NDRange<<std::endl;

    try {
		if(bVerbose) std::cout<<"Create CL Platform. Get list of OpenCL platforms."<<std::endl;
		
		cl::Platform::get(&platform);

		if (platform.empty()) {
			std::cerr << "OpenCL platforms not found." << std::endl;
			return false;
		}

		if(bVerbose) std::cout<<"Get Device. Get first available GPU device which supports double precision. "<<std::endl;
		for(auto p = platform.begin(); devices.empty() && p != platform.end(); p++) {
			std::vector<cl::Device> pldev;

			try {
				p->getDevices(CL_DEVICE_TYPE_GPU, &pldev);

				for(auto device = pldev.begin(); devices.empty() && device != pldev.end(); device++) {
					if (!device->getInfo<CL_DEVICE_AVAILABLE>()) continue;

					std::cout << "\t\tDevice Name: " << device->getInfo<CL_DEVICE_NAME>() << std::endl;
					std::cout << "\t\tDevice Vendor: " << device->getInfo<CL_DEVICE_VENDOR>() << std::endl;
      				std::cout << "\t\tDevice Version: " << device->getInfo<CL_DEVICE_VERSION>() << std::endl;

					std::string ext = device->getInfo<CL_DEVICE_EXTENSIONS>();

					if (
						ext.find("cl_khr_fp64") == std::string::npos &&
						ext.find("cl_amd_fp64") == std::string::npos
					) continue;

					devices.push_back(*device);
					context = cl::Context(devices);
				}
			} catch(...) {
				devices.clear();
			}
		}

		if (devices.empty()) {
			std::cerr << "GPUs with double precision not found." << std::endl;
			return false;
		}

		if(bVerbose) std::cout << devices[0].getInfo<CL_DEVICE_NAME>() << std::endl;
        queue = cl::CommandQueue(context, devices[0]);

		//if(bVerbose) std::cout<<"Create command queue. "<<std::endl;
		

		if(bVerbose) std::cout<<"Compile OpenCL program for found device. "<<std::endl;
		std::string shaderCodeStr;
		std::string fileName = "add.cl";
		std::string fullFileName = SHADER_PATH + fileName;
		readFile(fullFileName, shaderCodeStr);
        program = cl::Program(context, shaderCodeStr);		
		

		try {
			program.build(devices);
		} catch (const cl::Error&) {
			std::cerr
			<< "OpenCL compilation error" << std::endl
			<< program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0])
			<< std::endl;
			return false;
		}
    } catch (const cl::Error &err) {
		std::cerr
			<< "OpenCL error: "
			<< err.what() << "(" << err.err() << ")"
			<< std::endl;
		return false;
    }

    return true;
}


 bool CCLAPP::readFile(const std::string& filename, std::string &buffer) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
		std::cout<<"failed to open file: "<<filename<<std::endl;
        return false;
        //throw std::runtime_error("failed to open file!");
    }

    size_t fileSize = (size_t)file.tellg();
    buffer.resize(fileSize);

    file.seekg(0);
    file.read(&buffer[0], fileSize);

    file.close();

    return true;
}

#endif