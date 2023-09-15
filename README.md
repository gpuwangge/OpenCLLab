# OpenCLLab
![Windows](https://img.shields.io/badge/Windows-passing-brightgreen)
![Linux](https://img.shields.io/badge/Linux(X86)-passing-brightgreen)  

OpenCL code with C++ bindings for Windows  

## Build
mkdir build  
cd build  
cmake -G "MinGW Makefiles" ..   
make  

## Install
### Compiler
I use MinGW  
https://www.mingw-w64.org/downloads  
(Browse to the Sources section)  
Must choose posix version (x86_64-posix-seh) (8.1.0)  

### OpenCL
- Download SDK here:  
https://github.com/KhronosGroup/OpenCL-SDK  
I use v2023.04.17 (Synchronize with OpenCL v3.0.14 specification release): OpenCL-SDK-v2023.04.17-Win-x64.zip  
- Add include/ to environment variable INCLUDE.  
- Add lib/ to environment variable LIB.  

## How to Build Linux x86_64 binaries on Windows WSL
### Install WSL(Windows Subsystem for Linux)  
Open Windows PowerShell in administrator mode  
wsl --install  
or  
wsl --install -d Ubuntu  
(May need reboot during installation)  
Create admin account and password for WSL  
Update package with  
sudo apt update && sudo apt upgrade  
[optional] Map network drive \\wsl$  
sudo apt  install cmake  
(use cmake --version to check setup)   
sudo apt install g++  
### Build the Project on WSL
- Copy the project folder into the WSL file system  
- Install and load OpenCL for Linux x86-64  
sudo apt install opencl-headers ocl-icd-opencl-dev -y  
- Change CMakelists.txt  
remove the two lines that set compilers (Linux VS Code will call default gcc/g++ compilers)  
To find OpenCL lib, add these:  
find_package(OpenCL REQUIRED)  
link_libraries(OpenCL::OpenCL)  
(OpenCL is found here: \usr\lib\x86_64-linux-gnu\libOpenCL.so) 
- Open VS Code and connect to WSL, open the correct folder, then cmake .. and make   
- You may need change "\\" into "/" if your code has such  
### Other Useful Hints
- To check WSL version(in Windows PowerShell):  
wsl --list --verbose  
or  
wsl --status  
- To check Ubuntu version (in Windows PowerShell or Ubuntu):
lsb_release -a  
or  
cat /etc/os-release  
- To check GPU validation, use clinfo  
sudo apt install clinfo  
(By the time of this readme, WSL 2 has not officially supported OpenCL yet)  
- To check compiled binary info  
file filename  
- To build Linux ARM Binary
sudo apt-get install gcc-aarch64-linux-gnu  
sudo apt-get install g++-aarch64-linux-gnu  
(compilers are located in \usr\bin)  
Change CMakeLists.txt  
set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc)  
set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)  
(OpenCL lib can't be located this way though)  

### Alternative OpenCL Lib
If you install CUDA SDK, OpenCL is included in the CUDA SDK  
(You can use MinGW x86_64-win32-seh for CUDA version)  

## Credits
https://github.com/KhronosGroup/OpenCL-CLHPP/tree/main  
https://github.khronos.org/OpenCL-CLHPP/  
https://gist.github.com/ddemidov/2925717  
https://cnugteren.github.io/tutorial/pages/page1.html  
https://github.com/CNugteren/myGEMM/tree/e2a364537f2b8725b3f5ba5f81008d04558a2327  









