#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#include <vector>
#include <chrono>
#include <fstream>

#define BLOCK_SIZE 16

void horizontal_line_detection(unsigned char* input, unsigned char* output, int rows, int cols) {
    for (int row = 1; row < rows - 1; ++row) {
        for (int col = 1; col < cols - 1; ++col) {
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

            output[output_index] = (unsigned char) std::min(255, std::max(sum, 0));
        }
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

    std::vector<unsigned char> conv(rows * cols);
    cv::Mat m_out(rows, cols, CV_8UC1, conv.data());

    auto start = std::chrono::system_clock::now();
    
    horizontal_line_detection(gray, conv.data(), rows, cols);
    
    auto stop = std::chrono::system_clock::now();

	auto duration = stop - start;
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

	std::cout << ms << " ms" << std::endl;

    std::ofstream file("temps.txt", std::ios_base::app);
    file << "horizontal_line_cpp : " << ms << " ms\n" << std::endl;
    file.close();

    cv::imwrite(argv[2], m_out);

    return 0;
}
