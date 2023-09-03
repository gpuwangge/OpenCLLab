/**************
***
*** If you want to enable float64, use this code
***
**************/
/* 
#if defined(cl_khr_fp64)
#  pragma OPENCL EXTENSION cl_khr_fp64: enable
#elif defined(cl_amd_fp64)
#  pragma OPENCL EXTENSION cl_amd_fp64: enable
#else
#  error double precision is not supported
#endif
*/


kernel void add(
        ulong n,
        global const float *a, //double
        global const float *b, //double
        global float *c //double
        )
{
    size_t i = get_global_id(0);
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}