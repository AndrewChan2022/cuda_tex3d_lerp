#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <memory>
#include <vector>
#include <cassert>
#include <iostream>

static const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorName(error);
}

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

inline void __getLastCudaError(const char *errorMessage, const char *file,
                               const int line) {
  cudaError_t err = cudaGetLastError();

  if (cudaSuccess != err) {
    fprintf(stderr,
            "%s(%i) : getLastCudaError() CUDA error :"
            " %s : (%d) %s.\n",
            file, line, errorMessage, static_cast<int>(err),
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;


template <typename T = float>
class CUDATexture3D {
    static_assert(std::is_arithmetic_v<T>, "CUDATexture3D only supports arithmetic types.");
public:
    using Ptr = std::shared_ptr<CUDATexture3D>;

    // Constructor: Wraps the volume object and device buffer
    CUDATexture3D(size_t width, size_t height, size_t depth, const void* volumeData, 
        bool normalizedCoords, 
        cudaTextureFilterMode filter = cudaFilterModeLinear, 
        cudaTextureAddressMode addressMode = cudaAddressModeClamp)
        : width(width), height(height), depth(depth) 
    {
        // Create channel description for the template type T
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
        cudaExtent extent = make_cudaExtent(width, height, depth);

        // Allocate 3D array
        if (cudaMalloc3DArray(&cudaArray, &channelDesc, extent) != cudaSuccess) {
            throw std::runtime_error("Failed to allocate CUDA 3D array.");
        }

        // Copy data to the CUDA 3D array
        cudaMemcpy3DParms copyParams = {};
        copyParams.srcPtr = make_cudaPitchedPtr(const_cast<void*>(volumeData), width * sizeof(T), width, height);
        copyParams.dstArray = cudaArray;
        copyParams.extent = extent;
        copyParams.kind = cudaMemcpyHostToDevice;

        if (cudaMemcpy3D(&copyParams) != cudaSuccess) {
            cudaFreeArray(cudaArray);
            throw std::runtime_error("Failed to copy data to CUDA 3D array.");
        }

        // Create texture object
        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cudaArray;

        cudaTextureDesc texDesc = {};
        texDesc.addressMode[0] = addressMode;
        texDesc.addressMode[1] = addressMode;
        texDesc.addressMode[2] = addressMode;
        texDesc.filterMode = filter;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = normalizedCoords;

        if (cudaCreateTextureObject(&textureObject, &resDesc, &texDesc, nullptr) != cudaSuccess) {
            cudaFreeArray(cudaArray);
            throw std::runtime_error("Failed to create CUDA texture object.");
        }
    }

    // Destructor: Ensures resources are freed
    ~CUDATexture3D() {
        if (textureObject) {
            cudaDestroyTextureObject(textureObject);
        }
        if (cudaArray) {
            cudaFreeArray(cudaArray);
        }
    }

    // Delete copy constructor and copy assignment
    CUDATexture3D(const CUDATexture3D&) = delete;
    CUDATexture3D& operator=(const CUDATexture3D&) = delete;

    // Allow move constructor and move assignment
    CUDATexture3D(CUDATexture3D&& other) noexcept
        : cudaArray(other.cudaArray), textureObject(other.textureObject),
          width(other.width), height(other.height), depth(other.depth) 
    {
        other.cudaArray = nullptr;
        other.textureObject = 0;
    }

    CUDATexture3D& operator=(CUDATexture3D&& other) noexcept {
        if (this != &other) {
            // Clean up current resources
            if (textureObject) cudaDestroyTextureObject(textureObject);
            if (cudaArray) cudaFreeArray(cudaArray);

            // Transfer ownership
            cudaArray = other.cudaArray;
            textureObject = other.textureObject;
            width = other.width;
            height = other.height;
            depth = other.depth;

            other.cudaArray = nullptr;
            other.textureObject = 0;
        }
        return *this;
    }

    // Accessor for texture object
    cudaTextureObject_t getTextureObject() const { return textureObject; }

private:
    cudaArray_t cudaArray = nullptr;
    cudaTextureObject_t textureObject = 0;
    size_t width = 0, height = 0, depth = 0;

};


__device__ float LinearSampleVolume(cudaTextureObject_t volumeTex, float3 texCoord) {
    float v = tex3D<float>(volumeTex, texCoord.x, texCoord.y, texCoord.z);
    return v;
}

// Kernel to sample and print node values
//              width = 4
//              |-----o-----|-----o-----|-----o-----|-----o-----|
//      node    0           1           2           3           4   node count = 5, node max index = 4      // unnormalized coord
//      cell          0           1           2           3         cell count = 4, cell max index = 3      // normalied coord count
//      sig            -----------  ---------   ----------
//      seg                0            1           2               seg count = 3,  seg max index 2         // normalized corrd segment
//
//                    |                                    |
//      norm    0   0.125       0.325      0.625         0.875  1
//      unnorm  0    0.5                                  3.5   4
__global__ void tex1d_sample_kernel(cudaTextureObject_t tex, uint3 gridSize, uint3 dispatchSize) {
    // Compute thread-specific coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // if (x >= gridSize.x || y >= gridSize.y || z >= gridSize.z) return;

    float3 coord;
    coord.x = (x * 1.0) / (dispatchSize.x - 1);     // segment normalize 🌙
    // coord.x = coord.x = (x * 1.0) / 4.0;
    coord.y = 0.0;
    coord.z = 0.0;

    float dx = 1.0/ 8.0; //0.125 * 0.5;  🌙❤️ dx must be 2^n or 2^-n

    coord.x *= (gridSize.x - 1);                    // to int coord🌙
    coord.y = 0.0;
    coord.z = 0.0;

    // Unnormalized texture coordinates at texel center
    float3 texCoord = make_float3(coord.x + 0.5, coord.y + 0.5, coord.z + 0.5); // 🌙 int coord 0.5
    float3 offsetCoord = make_float3(coord.x + 0.5 + dx, coord.y, coord.z);

    // Sample at texCoord and offsetCoord
    float nodeValue = LinearSampleVolume(tex, texCoord);
    float offsetValue = LinearSampleVolume(tex, offsetCoord);

    // Print results
    printf("Node (%d, %d, %d) (%f %f %f): Value = %f, (+%f +0 +0) Offset Value = %f dv:%f\n", 
        x, y, z, 
        coord.x, coord.y, coord.z, nodeValue, 
        dx, offsetValue, offsetValue - nodeValue);
}

void launch_test_tex1d_sample() {

    // Define volume size
    constexpr size_t width = 4;
    constexpr size_t height = 1;
    constexpr size_t depth = 1;
    uint3 gridSize = make_uint3(width, height, depth);
    uint3 dispatchSize = make_uint3(width * 3 + 1, height * 1, depth * 1);
    cudaExtent volumeSize = make_cudaExtent(gridSize.x, gridSize.y, gridSize.z);

    // Allocate and initialize host volume data
    float *h_volume = new float[gridSize.x * gridSize.y * gridSize.z];
    auto h_grid = reinterpret_cast<float(*)[height][depth]>(h_volume);
    for (size_t z = 0; z < depth; z++) {
        for (size_t y = 0; y < height; y++) {
            for (size_t x = 0; x < width; x++) {
                h_grid[z][y][x] = x + y + z;
            }
        }
    }

    // Create texture object
    CUDATexture3D<float> texture(width, height, depth, h_volume, false);
    delete[] h_volume;

    // Launch kernel
    cudaTextureObject_t texObj = texture.getTextureObject();
    dim3 blockSize(8, 1, 1);
    dim3 gridSizeBlocks((dispatchSize.x + 7) / 8, (dispatchSize.y + 7) / 8, (dispatchSize.z + 7) / 8);
    tex1d_sample_kernel<<<gridSizeBlocks, blockSize>>>(texObj, gridSize, dispatchSize);
    cudaDeviceSynchronize();
}


// Kernel to sample and print node values
__global__ void tex1d_sample_norm_kernel(cudaTextureObject_t tex, uint3 gridSize, uint3 dispatchSize) {
    // Compute thread-specific coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // if (x >= gridSize.x || y >= gridSize.y || z >= gridSize.z) return;

    float3 coord;
    coord.x = (x * 1.0) / (dispatchSize.x - 1);
    // coord.x = coord.x = (x * 1.0) / 4.0;
    coord.y = 0.0;
    coord.z = 0.0;

    coord.x = (coord.x * (gridSize.x - 1) + 0.5) / gridSize.x;
    // coord.x *= (gridSize.x - 1);
    // coord.y = 0.0;
    // coord.z = 0.0;

    // coord.x = 0.25 + 0.125;

    // Unnormalized texture coordinates at texel center
    float3 texCoord = make_float3(coord.x, coord.y, coord.z);
    float dx = 1.0/ 32.0; //0.125 * 0.5;  🌙❤️ dx must be 2^n or 2^-n
    float3 offsetCoord = make_float3(coord.x + dx / gridSize.x, coord.y, coord.z);

    // Sample at texCoord and offsetCoord
    float nodeValue = LinearSampleVolume(tex, texCoord);
    float offsetValue = LinearSampleVolume(tex, offsetCoord);

    // Print results
    printf("Node (%d, %d, %d) (%f %f %f): Value = %f, (+%f +0 +0) Offset Value = %f dv:%f\n", x, y, z, 
        coord.x, coord.y, coord.z, nodeValue, 
        dx, offsetValue, offsetValue - nodeValue);
}

void launch_test_tex1d_sample_norm() {

    // Define volume size
    constexpr size_t width = 4;
    constexpr size_t height = 1;
    constexpr size_t depth = 1;
    uint3 gridSize = make_uint3(width, height, depth);
    uint3 dispatchSize = make_uint3(width * 3 + 1, height * 1, depth * 1);
    cudaExtent volumeSize = make_cudaExtent(gridSize.x, gridSize.y, gridSize.z);

    // Allocate and initialize host volume data
    float *h_volume = new float[gridSize.x * gridSize.y * gridSize.z];
    auto h_grid = reinterpret_cast<float(*)[height][depth]>(h_volume);
    for (size_t z = 0; z < depth; z++) {
        for (size_t y = 0; y < height; y++) {
            for (size_t x = 0; x < width; x++) {
                h_grid[z][y][x] = x + y + z;
            }
        }
    }

    // Create texture object
    CUDATexture3D<float> texture(width, height, depth, h_volume, true);
    delete[] h_volume;

    // Launch kernel
    cudaTextureObject_t texObj = texture.getTextureObject();
    dim3 blockSize(8, 1, 1);
    dim3 gridSizeBlocks((dispatchSize.x + 7) / 8, (dispatchSize.y + 7) / 8, (dispatchSize.z + 7) / 8);
    tex1d_sample_norm_kernel<<<gridSizeBlocks, blockSize>>>(texObj, gridSize, dispatchSize);
    cudaDeviceSynchronize();
}


int main() {
    std::cout << "Hello from C++!" << std::endl;
    
    launch_test_tex1d_sample();
    // launch_test_tex1d_sample_norm();

    return 0;
}
