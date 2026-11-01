#include "eps/device_features.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
namespace {
void Check(cudaError_t error){if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));}
__global__ void Gradient(const float* features,float* gradient,std::uint64_t count){
  for(std::uint64_t i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=std::uint64_t(blockDim.x)*gridDim.x)gradient[i]=2*features[i];
}
}
std::vector<float> DeviceModelGradient(const eps::DeviceTensorBlock& block){
  Check(cudaSetDevice(block.tensor->device()));const auto count=block.tensor->elements();std::vector<float> values(count);
  if(!count)return values;
  cudaStream_t stream=nullptr;float* gradient=nullptr;
  try{
    Check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));Check(cudaMalloc(&gradient,count*4));
    block.tensor->WaitOnStream(stream);
    Gradient<<<(count+255)/256,256,0,stream>>>(static_cast<const float*>(block.tensor->data()),gradient,count);Check(cudaGetLastError());
    Check(cudaMemcpyAsync(values.data(),gradient,count*4,cudaMemcpyDeviceToHost,stream));Check(cudaStreamSynchronize(stream));
  }catch(...){if(stream)cudaStreamSynchronize(stream);if(gradient)cudaFree(gradient);if(stream)cudaStreamDestroy(stream);throw;}
  cudaFree(gradient);cudaStreamDestroy(stream);return values;
}
