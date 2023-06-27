#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>

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

    auto rows = m_in.rows;
    auto cols = m_in.cols;

    cv::Mat m_out(rows, cols, CV_8UC1);

    // Start time measurement
    auto start = std::chrono::high_resolution_clock::now();

    for (int row = 1; row < rows - 1; ++row) {
        for (int col = 1; col < cols - 1; ++col) {

            int sum = 0;
            sum += -m_in.at<uchar>(row - 1, col - 1);
            sum += -m_in.at<uchar>(row - 1, col);
            sum += -m_in.at<uchar>(row - 1, col + 1);
            sum += -m_in.at<uchar>(row, col - 1);
            sum += 8 * m_in.at<uchar>(row, col);
            sum += -m_in.at<uchar>(row, col + 1);
            sum += -m_in.at<uchar>(row + 1, col - 1);
            sum += -m_in.at<uchar>(row + 1, col);
            sum += -m_in.at<uchar>(row + 1, col + 1);

            m_out.at<uchar>(row, col) = static_cast<unsigned char>(std::min(255, std::max(sum, 0)));
        }
    }

    // End time measurement
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "time=" << duration << "ms" << std::endl;

    cv::imwrite(argv[2], m_out);

    return 0;
}