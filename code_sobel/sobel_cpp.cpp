#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

#include <vector>
#include <chrono>
#include <fstream>

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
    int radius = 1;
    std::vector< unsigned char > g(height * width);
    cv::Mat output(height, width, CV_8UC1, g.data());

    auto start = std::chrono::system_clock::now();

    // Appliquer le filtre de flou
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int gx = 0, gy = 0;
            int i = x + y * width;

            if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
                gx = -1 * input.data[i - width - 1] + -2 * input.data[i - width] + -1 * input.data[i - width + 1] +
                    1 * input.data[i + width - 1] + 2 * input.data[i + width] + 1 * input.data[i + width + 1];
                gy = -1 * input.data[i - width - 1] + 1 * input.data[i - width + 1] +
                    -2 * input.data[i - 1] + 2 * input.data[i + 1] +
                    -1 * input.data[i + width - 1] + 1 * input.data[i + width + 1];
            }

            g[i] = (unsigned char)(sqrt(gx * gx + gy * gy) / 4.0f);
        }
    }

    auto stop = std::chrono::system_clock::now();

    auto duration = stop - start;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

    std::cout << ms << " ms" << std::endl;

    std::ofstream file("temps.txt", std::ios_base::app);
    file << "sobel_cpp : " << ms << " ms\n" << std::endl;
    file.close();

    // Enregistrer l'image de sortie
    cv::imwrite(argv[2], output);

    return 0;
}