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
    CCLAPP();
    ~CCLAPP();

    bool initDevice();
	void loadShader(std::string filename);
	bool buildProgram();

    bool bVerbose;
    const size_t maxNDRange = 1 << 20; //determine this value?

	cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;

	bool readFile(const std::string& filename, std::string &buffer);

private:
	std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;	
};

CCLAPP::CCLAPP(){}
CCLAPP::~CCLAPP(){}

bool CCLAPP::initDevice(){
    bVerbose = false;
	if(bVerbose) std::cout<<"NDRange: "<<maxNDRange<<std::endl;

    try {
		if(bVerbose) std::cout<<"Create CL Platform. Get list of OpenCL platforms."<<std::endl;
		
		cl::Platform::get(&platforms);

		if (platforms.empty()) {
			std::cerr << "OpenCL platforms not found." << std::endl;
			return false;
		}

		if(bVerbose) std::cout<<"Get Device. "<<std::endl;
		size_t i = 0;
		for(auto pPlatform = platforms.begin(); devices.empty() && pPlatform != platforms.end(); pPlatform++, i++) {
			std::cout << "Platform[" << i << "]:\n";
			PrintPlatformInfoSummary(*pPlatform);

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

		PrintDeviceInfoSummary(devices);

		if (devices.empty()) {
			std::cerr << "GPUs with double precision not found." << std::endl;
			return false;
		}

		if(bVerbose) std::cout << devices[0].getInfo<CL_DEVICE_NAME>() << std::endl;
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
	if(bVerbose) std::cout<<"Compile OpenCL program for found device. "<<std::endl;
	std::string shaderCodeStr;
	std::string fullFilename = SHADER_PATH + filename;
	readFile(fullFilename, shaderCodeStr);
	program = cl::Program(context, shaderCodeStr);	
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