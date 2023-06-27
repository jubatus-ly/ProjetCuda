#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <vector>
#include <chrono>
#include <fstream>

__global__ void laplacianOperatorShared(unsigned char* input, unsigned char* output, int rows, int cols)
{
    int row = blockIdx.y * (blockDim.y - 2) + threadIdx.y - 1;
    int col = blockIdx.x * (blockDim.x - 2) + threadIdx.x - 1;

    extern __shared__ unsigned char sh[];

    int sh_row = threadIdx.y;
    int sh_col = threadIdx.x;

    int sh_index = sh_row * (blockDim.x + 2) + sh_col;

    if (row >= 0 && col >= 0 && row < rows && col < cols)
    {
        sh[sh_index] = input[row * cols + col];
    }
    else
    {
        sh[sh_index] = 0;
    }

    __syncthreads();

    if (threadIdx.y > 0 && threadIdx.x > 0 && threadIdx.y < blockDim.y - 1 && threadIdx.x < blockDim.x - 1 && row >= 1 && col >= 1 && row < rows - 1 && col < cols - 1)
    {
        int output_index = row * cols + col;

        int sum = 0;
        sum += -sh[(sh_row - 1) * (blockDim.x + 2) + sh_col - 1];
        sum += -sh[(sh_row - 1) * (blockDim.x + 2) + sh_col];
        sum += -sh[(sh_row - 1) * (blockDim.x + 2) + sh_col + 1];
        sum += -sh[sh_row * (blockDim.x + 2) + sh_col - 1];
        sum += 8 * sh[sh_row * (blockDim.x + 2) + sh_col];
        sum += -sh[sh_row * (blockDim.x + 2) + sh_col + 1];
        sum += -sh[(sh_row + 1) * (blockDim.x + 2) + sh_col - 1];
        sum += -sh[(sh_row + 1) * (blockDim.x + 2) + sh_col];
        sum += -sh[(sh_row + 1) * (blockDim.x + 2) + sh_col + 1];

        output[output_index] = (unsigned char)min(255, max(sum, 0));
    }
}




int main(int argc, char** argv)
{
    if (argc != 3) {
        printf("Usage: ./programme <input_file> <output_file>\n");
        exit(-1);
    }

    cv::Mat m_in = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (m_in.empty()) {
        printf("Unable to load image '%s'\n", argv[1]);
        exit(-1);
    }

    auto gray = m_in.data;
    auto rows = m_in.rows;
    auto cols = m_in.cols;

    std::vector<unsigned char> lap(rows * cols);
    cv::Mat m_out(rows, cols, CV_8UC1, lap.data());

    cudaError_t cudaStatus;
    cudaError_t kernelStatus;

    unsigned char* gray_d;
    unsigned char* lap_d;

    cudaStatus = cudaMalloc(&gray_d, rows * cols);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error CudaMalloc gray_d: " << std::endl;
        exit(-1);
    }

    cudaStatus = cudaMalloc(&lap_d, rows * cols);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error CudaMalloc lap_d: " << std::endl;
        exit(-1);
    }

    dim3 block(32, 32);
    dim3 grid((cols - 1) / (block.x - 2) + 1, (rows - 1) / (block.y - 2) + 1);

    cudaEvent_t start, stop;
    cudaStatus = cudaEventCreate(&start);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaEventCreate start: " << std::endl;
        exit(-1);
    }

    cudaStatus = cudaEventCreate(&stop);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaEventCreate stop: " << std::endl;
        exit(-1);
    }

    cudaStatus = cudaMemcpy(gray_d, gray, rows * cols, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaMemcpy gray - HostToDevice: " << std::endl;
        exit(-1);
    }

    cudaStatus = cudaEventRecord(start);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaEventRecord start: " << std::endl;
        exit(-1);
    }

    laplacianOperatorShared<<<grid, block, (block.x + 2) * (block.y + 2) * sizeof(unsigned char)>>>(gray_d, lap_d, rows, cols);
    kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(kernelStatus) << std::endl;
        exit(-1);
    }

    cudaStatus = cudaEventRecord(stop);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaEventRecord stop: " << std::endl;
        exit(-1);
    }

    cudaStatus = cudaMemcpy(lap.data(), lap_d, rows * cols, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaMemcpy lap - DeviceToHost: " << std::endl;
        exit(-1);
    }

    cudaStatus = cudaEventSynchronize(stop);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaEventSynchronize: " << std::endl;
        exit(-1);
    }

    float duration;
    cudaStatus = cudaEventElapsedTime(&duration, start, stop);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaEventElapsedTime: " << std::endl;
        exit(-1);
    }

    std::cout << "time=" << duration << std::endl;

      cudaStatus=cudaEventDestroy(start);
       if (cudaStatus != cudaSuccess) 
  {
	std::cout << "Error CudaEvenDestroy start"  << " " ;
    exit(-1);

  }
   
    cudaStatus=cudaEventDestroy(stop);
     if (cudaStatus != cudaSuccess) 
  {
	std::cout << "Error CudaEvenDestroy stop"  << " " ;
    exit(-1);
  }

    cv::imwrite(argv[2], m_out);

      cudaStatus=cudaFree(gray_d);
     if (cudaStatus != cudaSuccess) 
  {
	std::cout << "Error Cudafree gray_d"  << " " ;
    exit(-1);
  }
    cudaStatus=cudaFree(lap_d);
    if (cudaStatus != cudaSuccess) 
  {
	std::cout << "Error Cudafree lap_d"  << " " ;
    exit(-1);
  }


    return 0;
}

