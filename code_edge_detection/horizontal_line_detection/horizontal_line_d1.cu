#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

#define BLOCK_SIZE 8


__global__ void horizontal_line_detection(unsigned char* input, unsigned char* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int row = idx / cols;
    int col = idx % cols;

    if (row >= 1 && col >= 1 && row < rows - 1 && col < cols - 1) {
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

        output[output_index] = (unsigned char) min(255,max(sum,0));
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

    unsigned char* gray_d;
    unsigned char* conv_d;

    
    cudaError_t err = cudaMalloc(&gray_d, rows * cols);
    if (err != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    err = cudaMalloc(&conv_d, rows * cols);
    if (err != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    
    err = cudaMemcpy(gray_d, gray, rows * cols, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    dim3 t(BLOCK_SIZE);
    int gridSize = (rows * cols + BLOCK_SIZE - 1) / BLOCK_SIZE;

    
    cudaEvent_t start, stop;
    err = cudaEventCreate(&start);
    if (err != cudaSuccess) {
        printf("cudaEventCreate failed: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) {
    printf("cudaEventCreate failed: %s\n", cudaGetErrorString(err));
    exit(-1);
}


err = cudaEventRecord(start);
if (err != cudaSuccess) {
    printf("cudaEventRecord failed: %s\n", cudaGetErrorString(err));
    exit(-1);
}

    horizontal_line_detection<<<gridSize, BLOCK_SIZE>>>(gray_d, conv_d, rows, cols);



err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
    exit(-1);
}


err = cudaEventRecord(stop);
if (err != cudaSuccess) {
    printf("cudaEventRecord failed: %s\n", cudaGetErrorString(err));
    exit(-1);
}

err = cudaMemcpy(conv.data(), conv_d, rows * cols, cudaMemcpyDeviceToHost);
if (err != cudaSuccess) {
    printf("cudaMemcpy failed: %s\n", cudaGetErrorString(err));
    exit(-1);
}


err = cudaEventSynchronize(stop);
if (err != cudaSuccess) {
    printf("cudaEventSynchronize failed: %s\n", cudaGetErrorString(err));
    exit(-1);
}
float duration;
err = cudaEventElapsedTime(&duration, start, stop);
if (err != cudaSuccess) {
    printf("cudaEventElapsedTime failed: %s\n", cudaGetErrorString(err));
    exit(-1);
}
std::cout << "time=" << duration << std::endl;

err = cudaEventDestroy(start);
if (err != cudaSuccess) {
    printf("cudaEventDestroy failed: %s\n", cudaGetErrorString(err));
    exit(-1);
}

err = cudaEventDestroy(stop);
if (err != cudaSuccess) {
    printf("cudaEventDestroy failed: %s\n", cudaGetErrorString(err));
    exit(-1);
}

std::ofstream file("temps.txt", std::ios_base::app);
file << "horizontal : " << duration << " ms\n" << std::endl;
file.close();

cv::imwrite(argv[2], m_out);

err=cudaFree(gray_d);
if (err != cudaSuccess) 
{
	std::cout << "Error Cudafree gray_d"  << " " ;
    exit(-1);
}
err=cudaFree(conv_d);
if (err != cudaSuccess) 
{
	std::cout << "Error Cudafree conv_d"  << " " ;
    exit(-1);
}

return 0;

}
