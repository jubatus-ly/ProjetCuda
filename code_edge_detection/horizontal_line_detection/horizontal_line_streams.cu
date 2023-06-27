#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

#define BLOCK_SIZE 4
__global__ void horizontal_line_detection(unsigned char* input, unsigned char* output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= 1 && col >= 1 && row < rows - 1 && col < cols - 1) {
        int output_index = row * cols + col;

        int sum = 0;
        sum += -input[(row - 1) * cols + col - 1];
        sum += -input[(row - 1) * cols + col];
        sum += -input[(row - 1) * cols + col + 1];
        sum += 2 * input[row * cols + col - 1];
        sum += 2 * input[row * cols + col];
        sum += 2 * input[row * cols + col + 1];
        sum += -input[(row + 1) * cols + col - 1];
        sum += -input[(row + 1) * cols + col];
        sum += -input[(row + 1) * cols + col + 1];

        output[output_index] = (unsigned char) min(255,max(sum,0));
    }
}

int main(int argc, char** argv)
{
       
    if (argc != 3) {
        printf("Usage: edge_detection <input_file> <output_file>\n");
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

    cudaError_t cudaStatus;
    cudaError_t kernelStatus;

    std::vector<unsigned char> conv(rows * cols);
    cv::Mat m_out(rows, cols, CV_8UC1, conv.data());

    unsigned char* gray_d;
    unsigned char* conv_d;

    cudaStatus = cudaMalloc(&gray_d, rows * cols);
    if(cudaStatus != cudaSuccess) {
        std::cout << "Error CudaMalloc gray_d" << std::endl;
        exit(-1);
    }

    cudaStatus = cudaMalloc(&conv_d, rows * cols);
    if(cudaStatus != cudaSuccess) {
        std::cout << "Error CudaMalloc conv_d: " <<std::endl;
        exit(-1);
    }


    dim3 t(BLOCK_SIZE, BLOCK_SIZE);
    dim3 b((cols - 1) / t.x + 1, (rows - 1) / t.y + 1);

    size_t inputSize = sizeof(unsigned char) * cols * rows;
    size_t outputSize = inputSize;
   
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

    
    cudaStatus = cudaEventRecord(start);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaEventRecord start: "  << std::endl;
        exit(-1);
    }

    
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

    cudaStatus = cudaMemcpyAsync(gray_d, gray, inputSize / 2, cudaMemcpyHostToDevice, stream1);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaMemcpyAsync gray - HostToDevice: "  << std::endl;
        exit(-1);
    }
    
    cudaStatus = cudaMemcpyAsync(gray_d + cols * (rows / 2), gray + cols * (rows / 2), inputSize / 2, cudaMemcpyHostToDevice, stream2);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaMemcpyAsync gray - HostToDevice: "  << std::endl;
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
    
    horizontal_line_detection<<< b, t, 0, stream1 >>>(gray_d, conv_d, rows/2, cols);
    kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(kernelStatus) << std::endl;
        exit(-1);
    }
    
    horizontal_line_detection<<< b, t, 0, stream2 >>>(gray_d + cols * (rows / 2), conv_d + cols * (rows / 2), rows/2, cols);
    kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(kernelStatus) << std::endl;
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
    
    cv::Mat output(rows, cols, CV_8UC1);
    cudaStatus = cudaMemcpyAsync(conv.data(), conv_d, outputSize, cudaMemcpyDeviceToHost, stream1);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaMemcpyAsync - DeviceToHost: "  << std::endl;
        exit(-1);
    }
    
    cudaStatus = cudaEventRecord(stop);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaEventRecord stop: " << std::endl;
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

    std::ofstream file("temps.txt", std::ios_base::app);
    file << "blur : " << duration << " ms\n" << std::endl;
    file.close();

    cv::imwrite(argv[2], m_out);

    cudaStatus=cudaFree(gray_d);
     if (cudaStatus != cudaSuccess) 
  {
	std::cout << "Error Cudafree gray_d"  << " " ;
    exit(-1);
  }
    cudaStatus=cudaFree(conv_d);
     if (cudaStatus != cudaSuccess) 
  {
	std::cout << "Error Cudafree conv_d"  << " " ;
    exit(-1);
  }
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

    return 0;
}
