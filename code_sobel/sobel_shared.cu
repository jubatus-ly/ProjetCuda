#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <vector>
#include <chrono>
#include <fstream>

#define BLOCK_SIZE 32

__global__ void sobel_shared( unsigned char* input, unsigned char* output, int width, int height )
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

    // Verification de la pr�sence des deux arguments
    if (argc != 3) {
        printf("Usage: edge_detection <input_file> <output_file>\n");
        exit(-1);
    }

    // Charger l'image d'entr�e en niveaux de gris
    cv::Mat input = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (input.empty()) {
        printf("Unable to load image '%s'\n", argv[1]);
        exit(-1);
    }

    // init cuda error
    cudaError_t cudaStatus;
    cudaError_t kernelStatus;

    // Allouer la m�moire pour les images d'entr�e et de sortie sur le GPU
    int width = input.cols;
    int height = input.rows;
    size_t inputSize = sizeof(unsigned char) * width * height;
    size_t outputSize = inputSize;
    unsigned char* d_input;
    unsigned char* d_output;
    cudaStatus = cudaMalloc(&d_input, inputSize);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaMalloc d_input: "  << std::endl;
        exit(-1);
    }
    cudaStatus = cudaMalloc(&d_output, outputSize);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaMalloc d_output: "  << std::endl;
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

    // Copier l'image d'entr�e sur le GPU
    cudaStatus = cudaMemcpy(d_input, input.data, inputSize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaMemcpy - HostToDevice: "  << std::endl;
        exit(-1);
    }

    dim3 block( BLOCK_SIZE, BLOCK_SIZE );
    /**
     * Pour la version shared il faut faire superposer les blocs de 2 pixels
     * pour ne pas avoir de bandes non calcul�es autour des blocs
     * on cr�e donc plus de blocs.
     */
    dim3 gridSize( ( width - 1) / (block.x-2) + 1 , ( height - 1 ) / (block.y-2) + 1 );

    // Version fusionn�e.
    sobel_shared<<< gridSize, block, block.x * block.y >>>( d_input, d_output, width, height );
    kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(kernelStatus) << std::endl;
        exit(-1);
    }
  
    // Copier l'image de sortie du GPU vers le CPU
    cv::Mat output(height, width, CV_8UC1);
    cudaStatus = cudaMemcpy(output.data, d_output, outputSize, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaMemcpy - DeviceToHost: "  << std::endl;
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

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::ofstream file("temps.txt", std::ios_base::app);
    file << "sobel_shared : " << duration << " ms\n" << std::endl;
    file.close();

    // Enregistrer l'image de sortie
    cv::imwrite(argv[2], output);

    // Lib�rer la m�moire allou�e sur le GPU
    cudaFree(d_input);
    cudaFree(d_output);
    
	return 0;
}