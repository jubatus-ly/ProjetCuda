#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <vector>
#include <chrono>
#include <fstream>


__global__ void laplacianOperator(unsigned char* input, unsigned char* output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row > 0 && col > 0 && row < rows - 1 && col < cols - 1) {
        int output_index = row * cols + col;

        int sum = 0;
        sum += -input[(row - 1) * cols + col - 1];
        sum += -input[(row - 1) * cols + col];
        sum += -input[(row - 1) * cols + col + 1];
        sum += -input[row * cols + col - 1];
        sum += 8 * input[row * cols + col];
        sum += -input[row * cols + col + 1];
        sum += -input[(row + 1) * cols + col - 1];
        sum += -input[(row + 1) * cols + col];
        sum += -input[(row + 1) * cols + col + 1];

        output[output_index] = (unsigned char) min(255,max(sum,0));
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

    auto rows = m_in.rows;
    auto cols = m_in.cols;

    cudaError_t cudaStatus;
    cudaError_t kernelStatus;

    std::vector<unsigned char> lap(rows * cols);
    cv::Mat m_out(rows, cols, CV_8UC1, lap.data());

    unsigned char* gray_d;
    unsigned char* lap_d;

    cudaStatus = cudaMalloc(&gray_d, rows * cols);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error CudaMalloc gray_d: "  << std::endl;
        exit(-1);
    }

    cudaStatus = cudaMalloc(&lap_d, rows * cols);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error CudaMalloc lap_d: "  << std::endl;
        exit(-1);
    }

    dim3 t(32, 32);
    dim3 b((cols - 1) / t.x + 1, (rows - 1) / t.y + 1);

    cudaStream_t stream1, stream2;
    cudaStatus = cudaStreamCreate(&stream1);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaStreamCreate stream1: "  << std::endl;
        exit(-1);
    }

    cudaStatus = cudaStreamCreate(&stream2);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaStreamCreate stream2: "  << std::endl;
        exit(-1);
    }

    cudaEvent_t start, stop;
    cudaStatus = cudaEventCreate(&start);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaEventCreate start: "  << std::endl;
        exit(-1);
    }

    cudaStatus = cudaEventCreate(&stop);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaEventCreate stop: "  << std::endl;
        exit(-1);
    }

    const int size = rows * cols;
    const int half_size = size / 2;

    cudaStatus = cudaMemcpyAsync(gray_d, m_in.data, half_size, cudaMemcpyHostToDevice, stream1);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaMemcpyAsync gray - HostToDevice: "  << std::endl;
        exit(-1);
    }

    cudaStatus = cudaMemcpyAsync(gray_d + half_size, m_in.data + half_size, size - half_size, cudaMemcpyHostToDevice, stream2);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaMemcpyAsync gray - HostToDevice: "  << std::endl;
        exit(-1);
    }

    cudaStatus = cudaEventRecord(start);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaEventRecord start: "  << std::endl;
        exit(-1);
    }

    laplacianOperator<<< b, t, 0, stream1 >>>(gray_d, lap_d, rows / 2 + 1, cols);
    kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(kernelStatus) << std::endl;
        exit(-1);
    }

    laplacianOperator<<< b, t, 0, stream2 >>>(gray_d + (rows / 2 - 1) * cols, lap_d + (rows / 2 - 1) * cols, rows - rows / 2 + 1, cols);
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

    cudaStatus = cudaMemcpyAsync(lap.data(), lap_d, half_size, cudaMemcpyDeviceToHost, stream1);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaMemcpyAsync lap - DeviceToHost: "  << std::endl;
        exit(-1);
    }

    cudaStatus = cudaMemcpyAsync(lap.data() + half_size, lap_d + half_size, size - half_size, cudaMemcpyDeviceToHost, stream2);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaMemcpyAsync lap - DeviceToHost: "  << std::endl;
        exit(-1);
    }

    cudaStatus = cudaStreamSynchronize(stream1);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaStreamSynchronize stream1: "  << std::endl;
        exit(-1);
    }

    cudaStatus = cudaStreamSynchronize(stream2);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaStreamSynchronize stream2: "  << std::endl;
        exit(-1);
    }

    cudaStatus = cudaEventSynchronize(stop);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaEventSynchronize: "  << std::endl;
        exit(-1);
    }

    float duration;
    cudaStatus = cudaEventElapsedTime(&duration, start, stop);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaEventElapsedTime: "  << std::endl;
        exit(-1);
    }

    std::cout << "time=" << duration << std::endl;

    cv::imwrite(argv[2], m_out);

      cudaStatus=cudaStreamDestroy(stream1);
     if (cudaStatus != cudaSuccess) 
  {
	std::cout << "Error CudaStreamDestroy stream1"  << " " ;
    exit(-1);

  }

    cudaStatus=cudaStreamDestroy(stream2);
      if (cudaStatus != cudaSuccess) 
  {
	std::cout << "Error CudaStreamDestroy stream1"  << " " ;
    exit(-1);

  }

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
