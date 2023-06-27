#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <vector>
#include <chrono>
#include <fstream>

#define BLOCK_SIZE 32

__global__ void sobelFilter(unsigned char* input, unsigned char* output, int width, int height) {
    // Calcul des coordonnées du thread en cours d'exécution
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Vérification que les coordonnées sont dans les limites de l'image
    if (x < width && y < height) {
        // Initialisation des variables de calcul du filtre
        int gx = 0, gy = 0;
        int i = x + y * width;

        // Vérification que les coordonnées du pixel sont dans les limites de l'image
        if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
            // Calcule les valeurs des gradients gx et gy en utilisant l'opérateur de Sobel
            gx = -1 * input[i - width - 1] + -2 * input[i - width] + -1 * input[i - width + 1] +
                 1 * input[i + width - 1] + 2 * input[i + width] + 1 * input[i + width + 1];
            gy = -1 * input[i - width - 1] + 1 * input[i - width + 1] +
                 -2 * input[i - 1] + 2 * input[i + 1] +
                 -1 * input[i + width - 1] + 1 * input[i + width + 1];
        }

        // Calcule la magnitude du gradient et stocke la valeur dans le tableau de sortie
        output[i] = (unsigned char) (__dsqrt_rn(gx * gx + gy * gy) / 4.0f);
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

    // Définir la taille des blocs et des grilles
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Appliquer le filtre de Sobel sur l'image d'entrée pour détecter les bordures
    sobelFilter<<<gridSize, blockSize>>>(d_input, d_output, width, height);
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
    file << "sobel : " << duration << " ms\n" << std::endl;
    file.close();

    // Enregistrer l'image de sortie
    cv::imwrite(argv[2], output);

    // Libérer la mémoire allouée sur le GPU
    cudaFree(d_input);
    cudaFree(d_output);
    
	return 0;
}