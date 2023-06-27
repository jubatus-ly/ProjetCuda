#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <vector>
#include <chrono>
#include <fstream>


__global__ void laplacianOperator(unsigned char* input, unsigned char* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int row = idx / cols;
    int col = idx % cols;

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


    auto gray = m_in.data;
    auto rows = m_in.rows;
    auto cols = m_in.cols;
    cudaError_t cudaStatus;
    cudaError_t kernelStatus;

    std::vector<unsigned char> lap(rows * cols);
    cv::Mat m_out(rows, cols, CV_8UC1, lap.data());

    unsigned char* gray_d;
    unsigned char* lap_d;

    cudaStatus=cudaMalloc(&gray_d, rows * cols);
    if (cudaStatus != cudaSuccess)
    {
        std::cout << "Error CudaMalloc gray_d"  << " ";
        exit(-1);

    }


    cudaStatus=cudaMalloc(&lap_d, rows * cols);
    if (cudaStatus != cudaSuccess)
  {
	std::cout << "Error CudaMalloc lap_d"  << " ";
    exit(-1);

  }



    int blockSize = 4;
    int gridSize = (rows * cols + blockSize - 1) / blockSize;

    // Creation des evenement pour le calcul du temps
    cudaEvent_t start, stop;

    cudaStatus=cudaEventCreate( &start );
     if (cudaStatus  != cudaSuccess)
  {
	  std::cout << "Error Eventcreate start" << " ";
      exit(-1);

  }
    cudaStatus=cudaEventCreate( &stop );
     if (cudaStatus  != cudaSuccess)
  {
	  std::cout << "Error Eventcreate stop" << " ";
      exit(-1);

    
  }

    cudaStatus=cudaMemcpy(gray_d, gray, rows * cols, cudaMemcpyHostToDevice);
    if (cudaStatus  != cudaSuccess)
  {
	  std::cout << "Error cudaMemcpy gray - HostToDevice" << " ";
      exit(-1);

  }


    cudaStatus=cudaEventRecord(start);
    if (cudaStatus  != cudaSuccess)
  {
	  std::cout << "Error cudaStart" << " ";
      exit(-1);

  }



    laplacianOperator<<<gridSize, blockSize>>>(gray_d, lap_d, rows, cols);
    kernelStatus = cudaGetLastError();
   if ( kernelStatus != cudaSuccess )
   {
	   std::cout << "CUDA Error"<< cudaGetErrorString(kernelStatus) << " ";
       exit(-1);

  }

    // End of computation time
   cudaStatus= cudaEventRecord(stop);
   if (cudaStatus  != cudaSuccess)
  {
	  std::cout << "Error cudaStop" << " ";
      exit(-1);

  }


   cudaStatus= cudaMemcpy(lap.data(), lap_d, rows * cols, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) 
  {
	std::cout << "Error Cuda Memcpy lap DeviceToHost"  << " " ;
     exit(-1);

  }

    // Calcul du temps total
   cudaStatus= cudaEventSynchronize( stop );
      if (cudaStatus != cudaSuccess) 
  {
	std::cout << "Error CudaEvenSychronize"  << " " ;
    exit(-1);

  }


    float duration;
    cudaEventElapsedTime( &duration, start, stop );
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
