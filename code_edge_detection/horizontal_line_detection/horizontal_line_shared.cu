#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

#define BLOCK_SIZE 32

__global__ void horizontal_line_detection_shared(unsigned char* input, unsigned char* output, int rows, int cols) {
    int row = blockIdx.y * (blockDim.y - 2) + threadIdx.y - 1;
    int col = blockIdx.x * (blockDim.x - 2) + threadIdx.x - 1;

    extern __shared__ unsigned char sh[];

    int sh_row = threadIdx.y;
    int sh_col = threadIdx.x;

    int sh_index = sh_row * (blockDim.x + 2) + sh_col;

    if (row >= 0 && col >= 0 && row < rows && col < cols) {
        sh[sh_index] = input[row * cols + col];
    } else {
        sh[sh_index] = 0;
    }

    __syncthreads();

    if (threadIdx.y > 0 && threadIdx.x > 0 && threadIdx.y < blockDim.y - 1 && threadIdx.x < blockDim.x - 1 && row >= 1 && col >= 1 && row < rows - 1 && col < cols - 1) {
        int output_index = row * cols + col;

        int sum = 0;
        sum += -sh[(sh_row - 1) * (blockDim.x + 2) + sh_col - 1];
        sum += -sh[(sh_row - 1) * (blockDim.x + 2) + sh_col];
        sum += -sh[(sh_row - 1) * (blockDim.x + 2) + sh_col + 1];
        sum += 2 * sh[sh_row * (blockDim.x + 2) + sh_col - 1];
        sum += 2 * sh[sh_row * (blockDim.x + 2) + sh_col];
        sum += 2 * sh[sh_row * (blockDim.x + 2) + sh_col + 1];
        sum += -sh[(sh_row + 1) * (blockDim.x + 2) + sh_col - 1];
        sum += -sh[(sh_row + 1) * (blockDim.x + 2) + sh_col];
        sum += -sh[(sh_row + 1) * (blockDim.x + 2) + sh_col + 1];

        output[output_index] = (unsigned char)min(255, max(sum, 0));
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
    printf("Failed to allocate memory for gray_d: %s\n", cudaGetErrorString(err));
    exit(-1);
    }

    err = cudaMalloc(&conv_d, rows * cols);
if (err != cudaSuccess) {
    printf("Failed to allocate memory for conv_d: %s\n", cudaGetErrorString(err));
    exit(-1);
}

err = cudaMemcpy(gray_d, gray, rows * cols, cudaMemcpyHostToDevice);
if (err != cudaSuccess) {
    printf("Failed to copy data from host to device: %s\n", cudaGetErrorString(err));
    exit(-1);
}

dim3 block(BLOCK_SIZE, BLOCK_SIZE);
dim3 grid((cols - 1) / (block.x - 2) + 1, (rows - 1) / (block.y - 2) + 1);

cudaEvent_t start, stop;
err = cudaEventCreate(&start);
if (err != cudaSuccess) {
    printf("Failed to create start event: %s\n", cudaGetErrorString(err));
    exit(-1);
}

err = cudaEventCreate(&stop);
if (err != cudaSuccess) {
    printf("Failed to create stop event: %s\n", cudaGetErrorString(err));
    exit(-1);
}

err = cudaEventRecord(start);
if (err != cudaSuccess) {
    printf("Failed to record start event: %s\n", cudaGetErrorString(err));
    exit(-1);
}

horizontal_line_detection_shared<<<grid, block, (block.x + 2) * (block.y + 2) * sizeof(unsigned char)>>>(gray_d, conv_d, rows, cols);

err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("Failed to launch kernel: %s\n", cudaGetErrorString(err));
    exit(-1);
}

err = cudaEventRecord(stop);
if (err != cudaSuccess) {
    printf("Failed to record stop event: %s\n", cudaGetErrorString(err));
    exit(-1);
}

err = cudaMemcpy(conv.data(), conv_d, rows * cols, cudaMemcpyDeviceToHost);
if (err != cudaSuccess) {
    printf("Failed to copy data from device to host: %s\n", cudaGetErrorString(err));
    exit(-1);
}

err = cudaEventSynchronize(stop);
if (err != cudaSuccess) {
    printf("Failed to synchronize stop event: %s\n", cudaGetErrorString(err));
    exit(-1);
}

float duration;
err = cudaEventElapsedTime(&duration, start, stop);
if (err != cudaSuccess) {
    printf("Failed to get elapsed time: %s\n", cudaGetErrorString(err));
    exit(-1);
}
std::cout << "time=" << duration << std::endl;
err = cudaEventDestroy(start);
if (err != cudaSuccess) 
{
	std::cout << "Error CudaEvenDestroy start"  << " " ;
    exit(-1);

}
err = cudaEventDestroy(stop);
     if (err != cudaSuccess) 
  {
	std::cout << "Error CudaEvenDestroy stop"  << " " ;
    exit(-1);

  }
std::ofstream file("temps.txt", std::ios_base::app);
file << "horizontal_shared : " << duration << " ms\n" << std::endl;
file.close();

cv::imwrite(argv[2], m_out);

err = cudaFree(gray_d);
if (err != cudaSuccess) 
{
	std::cout << "Error Cudafree gray_d"  << " " ;
    exit(-1);
}
err = cudaFree(conv_d);
if (err != cudaSuccess) 
{
	std::cout << "Error Cudafree conv_d"  << " " ;
    exit(-1);
}
return 0;
}
