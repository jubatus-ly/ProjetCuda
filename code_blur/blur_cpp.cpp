#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

#include <vector>
#include <chrono>
#include <fstream>

#define RADIUS 5

int main(int argc, char** argv)
{
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

    // Creation de la sortie
    int width = input.cols;
    int height = input.rows;
    int radius = RADIUS;
	std::vector< unsigned char > g(height * width);
	cv::Mat output(height, width, CV_8UC1, g.data());

	auto start = std::chrono::system_clock::now();

    // Appliquer le filtre de flou
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int sum = 0;
            int count = 0;
            for (int j = -radius; j <= radius; j++) {
                for (int i = -radius; i <= radius; i++) {
                    int offsetX = x + i;
                    int offsetY = y + j;
                    if (offsetX >= 0 && offsetX < width && offsetY >= 0 && offsetY < height) {
                        sum += input.data[offsetX + offsetY * width];
                        count++;
                    }
                }
            }
            g[x + y * width] = sum / count;
        }
    }

	auto stop = std::chrono::system_clock::now();

	auto duration = stop - start;
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

	std::cout << ms << " ms" << std::endl;

    std::ofstream file("temps.txt", std::ios_base::app);
    file << "blur_cpp : " << ms << " ms\n" << std::endl;
    file.close();

    // Enregistrer l'image de sortie
    cv::imwrite(argv[2], output);

	return 0;
}