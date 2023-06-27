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
    // Calcul des coordonnées du thread en cours d'exécution
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Vérification que les coordonnées sont dans les limites de l'image
    if (index < width * height) {
        int x = index % width;
        int y = index / width;

        // Initialisation des variables de calcul du flou
        int sum = 0;
        int count = 0;

        // Parcours de la zone de flou
        for (int i = -radius; i <= radius; i++) {
            for (int j = -radius; j <= radius; j++) {
                // Calcul des coordonnées du pixel à ajouter au calcul du flou
                int offsetX = x + i;
                int offsetY = y + j;

                // Vérification que les coordonnées du pixel sont dans les limites de l'image
                if (offsetX >= 0 && offsetX < width && offsetY >= 0 && offsetY < height) {
                    // Ajout du pixel au calcul du flou
                    sum += input[offsetX + offsetY * width];
                    count++;
                }
            }
        }
        // Calcul de la valeur moyenne des pixels dans la zone de flou
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

    // Copier l'image d'entrée sur le GPU
    cudaStatus = cudaMemcpy(d_input, input.data, inputSize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaMemcpy - HostToDevice: "  << std::endl;
        exit(-1);
    }

    // Définir la taille des blocs et des grilles
    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize((width*height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Appliquer le filtre de flou sur l'image d'entrée
    boxBlurFilter<<<gridSize, blockSize>>>(d_input, d_output, width, height, radius);
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
    file << "blur1D : " << duration << " ms\n" << std::endl;
    file.close();

    // Enregistrer l'image de sortie
    cv::imwrite(argv[2], output);

    // Libérer la mémoire allouée sur le GPU
    cudaFree(d_input);
    cudaFree(d_output);
    
	return 0;
}