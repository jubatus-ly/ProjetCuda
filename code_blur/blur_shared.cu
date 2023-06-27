#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <vector>
#include <chrono>
#include <fstream>

#define BLOCK_SIZE 32
#define RADIUS 5

__global__ void boxBlurFilter_shared(unsigned char* input, unsigned char* output, int width, int height, int radius) {
    int x = blockIdx.x * (blockDim.x - 2 * radius) + threadIdx.x;
    int y = blockIdx.y * (blockDim.y - 2 * radius) + threadIdx.y;

    extern __shared__ unsigned char sh[];

    if(x < width && y < height) {
        sh[threadIdx.y * blockDim.x + threadIdx.x] = input[y * width + x];
    }

    __syncthreads();

    if(threadIdx.x >= radius && threadIdx.x < (blockDim.x - radius) && threadIdx.y >= radius && threadIdx.y < (blockDim.y - radius)) {
        int sum = 0;
        int count = 0;
        for(int i = -radius; i <= radius; i++) {
            for(int j = -radius; j <= radius; j++) {
                sum += sh[(threadIdx.y + i) * blockDim.x + threadIdx.x + j];
                count++;
            }
        }
        if(x < width && y < height) {
            output[y * width + x] = sum / count;
        }
    }
}

int main(int argc, char** argv) {

    // Verification de la présence des deux arguments
    if (argc != 3) {
        printf("Usage: edge_detection <input_file> <output_file>\n");
        exit(-1);
    }

    // Charger l'image d'entrée en niveaux de gris
    cv::Mat input = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (input.empty()) {
        printf("Unable to load image '%s'\n", argv[1]);
        exit(-1);
    }

    // init cuda error
    cudaError_t cudaStatus;
    cudaError_t kernelStatus;

    // Allouer la mémoire pour les images d'entrée et de sortie sur le GPU
    int width = input.cols;
    int height = input.rows;
    int radius = RADIUS;
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

    // Copier l'image d'entrée sur le GPU
    cudaStatus = cudaMemcpy(d_input, input.data, inputSize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaMemcpy - HostToDevice: "  << std::endl;
        exit(-1);
    }

    dim3 blockSize( BLOCK_SIZE, BLOCK_SIZE );
    /**
     * Pour la version shared il faut faire superposer les blocs de 2 pixels
     * pour ne pas avoir de bandes non calculées autour des blocs
     * on crée donc plus de blocs.
     */
    dim3 gridSize((width + BLOCK_SIZE - 1 - 2 * radius) / (BLOCK_SIZE - 2 * radius), (height + BLOCK_SIZE - 1 - 2 * radius) / (BLOCK_SIZE - 2 * radius));

    // Version fusionnée.
    boxBlurFilter_shared<<< gridSize, blockSize, blockSize.x * blockSize.y * sizeof(unsigned char) >>>( d_input, d_output, width, height, radius);
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
    file << "blur_shared : " << duration << " ms\n" << std::endl;
    file.close();

    // Enregistrer l'image de sortie
    cv::imwrite(argv[2], output);

    // Libérer la mémoire allouée sur le GPU
    cudaFree(d_input);
    cudaFree(d_output);
    
	return 0;
}