#pragma once
#include <cstddef>
#include <cstdint>
#define __host__
#define __device__
#define __global__
#define __forceinline__ inline
#define __restrict__
#define __launch_bounds__(...)
typedef int cudaError_t;
typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;
struct float2 { float x, y; };
struct float4 { float x, y, z, w; };
struct int2 { int x, y; };
struct int4 { int x, y, z, w; };
struct uint2 { unsigned x, y; };
struct uint4 { unsigned x, y, z, w; };
struct dim3 { unsigned x, y, z; dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {} };
static const cudaError_t cudaSuccess = 0;
inline cudaError_t cudaGetLastError() { return 0; }
inline const char* cudaGetErrorString(cudaError_t) { return ""; }
inline cudaError_t cudaDeviceSynchronize() { return 0; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return 0; }
inline cudaError_t cudaMemcpyAsync(void*, const void*, size_t, int, cudaStream_t = nullptr) { return 0; }
inline cudaError_t cudaMemsetAsync(void*, int, size_t, cudaStream_t = nullptr) { return 0; }
inline cudaError_t cudaStreamCreate(cudaStream_t*) { return 0; }
inline cudaError_t cudaStreamDestroy(cudaStream_t) { return 0; }
inline cudaError_t cudaEventCreate(cudaEvent_t*) { return 0; }
inline cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t = nullptr) { return 0; }
inline cudaError_t cudaEventSynchronize(cudaEvent_t) { return 0; }
inline cudaError_t cudaEventElapsedTime(float*, cudaEvent_t, cudaEvent_t) { return 0; }
inline cudaError_t cudaDeviceGetAttribute(int*, int, int) { return 0; }
inline cudaError_t cudaGetDevice(int*) { return 0; }
inline cudaError_t cudaSetDevice(int) { return 0; }
inline cudaError_t cudaMalloc(void**, size_t) { return 0; }
inline cudaError_t cudaFree(void*) { return 0; }
inline cudaError_t cudaHostAlloc(void**, size_t, unsigned) { return 0; }
inline cudaError_t cudaFreeHost(void*) { return 0; }
inline cudaError_t cudaMemGetInfo(size_t*, size_t*) { return 0; }

enum cudaLaunchAttributeID { cudaLaunchAttributeProgrammaticStreamSerialization = 1, cudaLaunchAttributePreferredSharedMemoryCarveout = 2 };
struct cudaLaunchAttributeValue { int programmaticStreamSerializationAllowed; int preferredSharedMemoryCarveout; int sharedMemCarveout; };
struct cudaLaunchAttribute { cudaLaunchAttributeID id; cudaLaunchAttributeValue val; };
struct cudaLaunchConfig_t { dim3 gridDim; dim3 blockDim; size_t dynamicSmemBytes; cudaStream_t stream; cudaLaunchAttribute* attrs; unsigned numAttrs; };
inline cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t*, const void*, void**) { return 0; }
inline cudaError_t cudaFuncSetAttribute(const void*, int, int) { return 0; }
inline cudaError_t cudaStreamBeginCapture(cudaStream_t, int) { return 0; }
inline cudaError_t cudaStreamEndCapture(cudaStream_t, void**) { return 0; }
inline cudaError_t cudaGraphInstantiate(void**, void*, unsigned long long) { return 0; }
inline cudaError_t cudaGraphLaunch(void*, cudaStream_t) { return 0; }
inline cudaError_t cudaGraphDestroy(void*) { return 0; }
inline cudaError_t cudaGraphExecDestroy(void*) { return 0; }
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t*, unsigned) { return 0; }
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t*, unsigned) { return 0; }
inline cudaError_t cudaDeviceSetLimit(int, size_t) { return 0; }
inline cudaError_t cudaDeviceGetLimit(size_t*, int) { return 0; }
inline cudaError_t cudaMemcpy(void*, const void*, size_t, int) { return 0; }
inline cudaError_t cudaMemset(void*, int, size_t) { return 0; }
inline cudaError_t cudaHostRegister(void*, size_t, unsigned) { return 0; }
inline cudaError_t cudaHostUnregister(void*) { return 0; }
inline cudaError_t cudaPointerGetAttributes(void*, const void*) { return 0; }
inline cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned) { return 0; }
inline cudaError_t cudaEventQuery(cudaEvent_t) { return 0; }
inline cudaError_t cudaDeviceReset() { return 0; }

