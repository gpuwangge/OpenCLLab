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

#include "utility.h"

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
    CCLAPP(bool verbose, bool profiler, bool verify);
    ~CCLAPP();

    bool initDevice();
	void loadShader(std::string filename);
	bool buildProgram();

    bool bVerbose;
	bool bProfiler;
	bool bVerify;
    const size_t maxNDRange = 1 << 20; //1048576, determine this value?

	cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;

	bool readFile(const std::string& filename, std::string &buffer);

private:
	std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;	
};

CCLAPP::CCLAPP(bool verbose, bool profiler, bool verify){
	bVerbose = verbose;
	bProfiler = profiler;
	bVerify = verify;
}
CCLAPP::~CCLAPP(){}

bool CCLAPP::initDevice(){
	if(bVerbose) std::cout<<"NDRange: "<<maxNDRange<<std::endl;

    try {
		cl::Platform::get(&platforms);

		if (platforms.empty()) {
			std::cerr << "OpenCL platforms not found." << std::endl;
			return false;
		}

		size_t i = 0;
		for(auto pPlatform = platforms.begin(); devices.empty() && pPlatform != platforms.end(); pPlatform++, i++) {
			if(bProfiler) std::cout << "Platform[" << i << "]:\n";
			if(bProfiler) PrintPlatformInfoSummary(*pPlatform);

			std::vector<cl::Device> pldev;

			try {
				pPlatform->getDevices(CL_DEVICE_TYPE_GPU, &pldev);

				for(auto device = pldev.begin(); devices.empty() && device != pldev.end(); device++) {
					if (!device->getInfo<CL_DEVICE_AVAILABLE>()) continue;

					std::string ext = device->getInfo<CL_DEVICE_EXTENSIONS>();

					// Get first available GPU device which supports double precision.
					/*
					if (
						ext.find("cl_khr_fp64") == std::string::npos &&
						ext.find("cl_amd_fp64") == std::string::npos
					) continue;
					*/

					devices.push_back(*device);
					context = cl::Context(devices);
				}
			} catch(...) {
				devices.clear();
			}
		}

		if(bProfiler) PrintDeviceInfoSummary(devices);

		if (devices.empty()) {
			std::cerr << "GPUs with double precision not found." << std::endl;
			return false;
		}

        queue = cl::CommandQueue(context, devices[0]);

		//if(bVerbose) std::cout<<"Create command queue. "<<std::endl;
    } catch (const cl::Error &err) {
		std::cerr
			<< "OpenCL error: "
			<< err.what() << "(" << err.err() << ")"
			<< std::endl;
		return false;
    }

    return true;
}

void CCLAPP::loadShader(std::string filename){
	std::string shaderCodeStr;
	std::string fullFilename = SHADER_PATH + filename;
	readFile(fullFilename, shaderCodeStr);
	program = cl::Program(context, shaderCodeStr);	
	if(bVerbose) std::cout<<"Compile OpenCL program: "<<fullFilename<<std::endl;
}

bool CCLAPP::buildProgram(){
	try {
		program.build(devices);
	} catch (const cl::Error&) {
		std::cerr
		<< "OpenCL compilation error" << std::endl
		<< program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0])
		<< std::endl;
		return false;
	}
	return true;
}

/**************
***
*** Utility Functions
***
**************/

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