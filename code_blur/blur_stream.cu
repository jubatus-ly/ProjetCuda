#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <vector>
#include <chrono>
#include <fstream>

#define BLOCK_SIZE 32
#define RADIUS 5

__global__ void boxBlurFilter(unsigned char* input, unsigned char* output, int width, int height, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int sum = 0;
        int count = 0;
        for (int i = -radius; i <= radius; i++) {
            for (int j = -radius; j <= radius; j++) {
                int offsetX = x + i;
                int offsetY = y + j;
                if (offsetX >= 0 && offsetX < width && offsetY >= 0 && offsetY < height) {
                    sum += input[offsetX + offsetY * width];
                    count++;
                }
            }
        }
        output[x + y * width] = sum / count;
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

    // Définir la taille des blocs et des grilles
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Copier la première moitié de l'image d'entrée sur le GPU avec le premier stream
    cudaStatus = cudaMemcpyAsync(d_input, input.data, inputSize / 2, cudaMemcpyHostToDevice, stream1);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaMemcpyAsync stream1 - HostToDevice: "  << std::endl;
        exit(-1);
    }

    // Copier la deuxième moitié de l'image d'entrée sur le GPU avec le deuxième stream
    cudaStatus = cudaMemcpyAsync(d_input + width * (height / 2), input.data + width * (height / 2), inputSize / 2, cudaMemcpyHostToDevice, stream2);
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

    // Appliquer le filtre de flou sur la première moitié de l'image d'entrée avec le premier stream
    boxBlurFilter<<<gridSize, blockSize, 0, stream1>>>(d_input, d_output, width, height / 2, radius);
    kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(kernelStatus) << std::endl;
        exit(-1);
    }

    // Appliquer le filtre de flou sur la deuxième moitié de l'image d'entrée avec le deuxième stream
    boxBlurFilter<<<gridSize, blockSize, 0, stream2>>>(d_input + width * (height / 2), d_output + width * (height / 2), width, height / 2, radius);
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
    cv::Mat output(height, width, CV_8UC1);
    cudaStatus = cudaMemcpyAsync(output.data, d_output, outputSize, cudaMemcpyDeviceToHost, stream1);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaMemcpyAsync stream1 - DeviceToHost: "  << std::endl;
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
    file << "blur_stream : " << duration << " ms\n" << std::endl;
    file.close();

    // Enregistrer l'image de sortie
    cv::imwrite(argv[2], output);

    // Libérer la mémoire allouée sur le GPU
    cudaFree(d_input);
    cudaFree(d_output);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
	return 0;
}