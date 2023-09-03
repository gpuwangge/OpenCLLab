﻿# OpenCLLab
OpenCL code with C++ Wrapper for Windows

# Build
mkdir build  
cd build  
cmake -G "MinGW Makefiles" ..   
make  

# Install
## Compiler
I use MinGW  
https://www.mingw-w64.org/downloads  
(Browse to the Sources section)  
You must use posix version (x86_64-posix-seh) (8.1.0)  

## OpenCL
Download SDK here:  
https://github.com/KhronosGroup/OpenCL-SDK  
I use v2023.04.17 (Synchronize with OpenCL v3.0.14 specification release): OpenCL-SDK-v2023.04.17-Win-x64.zip  
Add include/ to environment variable INCLUDE.  
Add lib/ to environment variable LIB.  

## Alternative
If you install CUDA SDK, OpenCL is included in the CUDA SDK  
(You can use MinGW x86_64-win32-seh for CUDA version)  








