#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <vector>
#include <chrono>
#include <fstream>

#define BLOCK_SIZE 32

__global__ void sobelFilter( unsigned char* input, unsigned char* output, int width, int height )
{
  auto li = threadIdx.x;
  auto lj = threadIdx.y;

  auto w = blockDim.x;
  auto h = blockDim.y;

  auto i = blockIdx.x * (blockDim.x-2) + threadIdx.x;
  auto j = blockIdx.y * (blockDim.y-2) + threadIdx.y;

  extern __shared__ unsigned char sh[];

  if( i < width && j < height )
  {
    sh[ lj * w + li ] = input[ j * width + i ];
  }

  __syncthreads();

  if( i < width -1 && j < height-1 && li > 0 && li < (w-1) && lj > 0 && lj < (h-1) )
  {
    auto gx =     sh[ (lj-1)*w + li - 1 ] -     sh[ (lj-1)*w + li + 1 ]
           + 2 * sh[ (lj  )*w + li - 1 ] - 2 * sh[ (lj  )*w + li + 1 ]
           +     sh[ (lj+1)*w + li - 1 ] -     sh[ (lj+1)*w + li + 1 ];

    auto gy =     sh[ (lj-1)*w + li - 1 ] -     sh[ (lj+1)*w + li - 1 ]
           + 2 * sh[ (lj-1)*w + li     ] - 2 * sh[ (lj+1)*w + li     ]
           +     sh[ (lj-1)*w + li + 1 ] -     sh[ (lj+1)*w + li + 1 ];

    auto res = gx*gx + gy*gy;

    output[ j * width + i ] = sqrtf( res );
  }
}

int main(int argc, char** argv) {
    // Verification de la présence des deux arguments
    if (argc != 3) {
        printf("Usage: edge_detection <input_file> <output_file>\n");
        exit(-1);
    }

    // Charger l'image d'entrée en niveaux de gris
    cv::Mat m_in = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (m_in.empty()) {
        printf("Unable to load image '%s'\n", argv[1]);
        exit(-1);
    }

    // init cuda error
    cudaError_t cudaStatus;
    cudaError_t kernelStatus;

    // Allouer la mémoire pour les images d'entrée et de sortie sur le GPU
    int width = m_in.cols;
    int height = m_in.rows;
    int half_height = height / 2;
    size_t inputSize1 = sizeof(unsigned char) * width * (half_height + 1);
    size_t inputSize2 = sizeof(unsigned char) * width * (height - half_height + 1);
    unsigned char* d_input1;
    unsigned char* d_input2;
    unsigned char* d_output1;
    unsigned char* d_output2;
    cudaStatus = cudaMalloc(&d_input1, inputSize1);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaMalloc d_input1: "  << std::endl;
        exit(-1);
    }
    cudaStatus = cudaMalloc(&d_input2, inputSize2);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaMalloc d_input2: "  << std::endl;
        exit(-1);
    }
    cudaStatus = cudaMalloc(&d_output1, inputSize1);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaMalloc d_output1: "  << std::endl;
        exit(-1);
    }
    cudaStatus = cudaMalloc(&d_output2, inputSize2);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaMalloc d_output2: "  << std::endl;
        exit(-1);
    }

    // Creation des evenement pour le calcul du temps
    cudaEvent_t start, stop;
    cudaStatus = cudaEventCreate(&start);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaEventCreate start: "  << std::endl;
        exit(-1);
    }
    cudaStatus = cudaEventCreate(&stop);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaEventCreate stop: " << std::endl;
        exit(-1);
    }

    // Mesure du temps de calcul du kernel uniquement
    cudaEventRecord( start );

    // Créer deux streams CUDA
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

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize1( ( width - 1) / (block.x-2) + 1 , ( half_height + 1 - 1 ) / (block.y-2) + 1 );
    dim3 gridSize2( ( width - 1) / (block.x-2) + 1 , ( height - half_height + 1 - 1 ) / (block.y-2) + 1 );

    // Copier la première moitié de l'image d'entrée sur le GPU avec le premier stream
    cudaStatus = cudaMemcpyAsync(d_input1, m_in.data, inputSize1, cudaMemcpyHostToDevice, stream1);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaMemcpyAsync stream1 - HostToDevice: "  << std::endl;
        exit(-1);
    }

    // Copier la deuxième moitié de l'image d'entrée sur le GPU avec le deuxième stream
    cudaStatus = cudaMemcpyAsync(d_input2, m_in.data + half_height * width - 1 * width, inputSize2, cudaMemcpyHostToDevice, stream2);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaMemcpyAsync stream2 - HostToDevice: "  << std::endl;
        exit(-1);
    }

    // Synchroniser les streams
    cudaStatus = cudaStreamSynchronize(stream1);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaStreamSynchronize stream1: " << std::endl;
        exit(-1);
    }
    cudaStatus = cudaStreamSynchronize(stream2);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaStreamSynchronize stream2: "  << std::endl;
        exit(-1);
    }

    // Appliquer le filtre de sobel sur la première moitié de l'image d'entrée avec le premier stream
    sobelFilter<<< gridSize1, block, block.x * block.y, stream1 >>>( d_input1, d_output1, width, half_height + 1);
    kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(kernelStatus) << std::endl;
        exit(-1);
    }

    // Appliquer le filtre de sobel sur la deuxième moitié de l'image d'entrée avec le deuxième stream
    sobelFilter<<< gridSize2, block, block.x * block.y, stream2 >>>( d_input2, d_output2, width, height - half_height + 1);
    kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(kernelStatus) << std::endl;
        exit(-1);
    }

    // Synchroniser les streams
    cudaStatus = cudaStreamSynchronize(stream1);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaStreamSynchronize stream1: " << std::endl;
        exit(-1);
    }
    cudaStatus = cudaStreamSynchronize(stream2);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaStreamSynchronize stream2: "  << std::endl;
        exit(-1);
    }

    // Copier l'image de sortie du GPU vers le CPU avec le premier stream
    cv::Mat output1(half_height + 1, width, CV_8UC1);
    cv::Mat output2(height - half_height + 1, width, CV_8UC1);

    cudaStatus = cudaMemcpyAsync(output1.data, d_output1, inputSize1, cudaMemcpyDeviceToHost, stream1);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaMemcpyAsync stream1 - DeviceToHost: "  << std::endl;
        exit(-1);
    }
    cudaStatus = cudaMemcpyAsync(output2.data, d_output2, inputSize2, cudaMemcpyDeviceToHost, stream2);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaMemcpyAsync stream2 - DeviceToHost: "  << std::endl;
        exit(-1);
    }

    // Fin du temps de calcul
    cudaStatus = cudaEventRecord( stop );
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaEventRecord stop : "  << std::endl;
        exit(-1);
    }

    // Calcul du temps total
    cudaStatus = cudaEventSynchronize(stop);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaEventSynchronize stop : "  << std::endl;
        exit(-1);
    }
    float duration;
    cudaEventElapsedTime( &duration, start, stop );
    std::cout << "time=" << duration << std::endl;

    std::ofstream file("temps.txt", std::ios_base::app);
    file << "sobel_shared_stream : " << duration << " ms\n" << std::endl;
    file.close();

    // Enregistrer l'image de sortie
    cv::Mat output(height, width, CV_8UC1);

    cv::Mat im1(output, cv::Rect(0, 0, width, half_height));
    output1(cv::Rect(0, 0, width, half_height)).copyTo(im1);

    cv::Mat im2(output, cv::Rect(0, half_height, width, height - half_height));
    output2(cv::Rect(0, 1, width, height - half_height)).copyTo(im2);

    cv::imwrite(argv[2], output);

    // Libérer la mémoire allouée sur le GPU
    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_output1);
    cudaFree(d_output2);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
	return 0;
}
