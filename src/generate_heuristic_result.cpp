#include <iostream>
#include <cudnn.h>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

int main() {
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t inputDesc, outputDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;

    // Create cuDNN handle
    cudnnCreate(&handle);

    std::ifstream infile("generate_models_configs.csv");
    std::ofstream outfile("dataset_heuristic.csv");
    std::string line;
    std::getline(infile, line);  // Skip header line

    // Write header to dataset
    outfile << line << ",label\n";

    int num_iterations = 30000;  // Set your desired number of iterations here
    for (int iter = 0; iter < num_iterations && std::getline(infile, line); ++iter) {
        std::istringstream iss(line);
        std::string token;

        // Parse CSV line
        std::getline(iss, token, ','); int input_n = std::stoi(token);
        std::getline(iss, token, ','); int input_c = std::stoi(token);
        std::getline(iss, token, ','); int input_h = std::stoi(token);
        std::getline(iss, token, ','); int input_w = std::stoi(token);
        std::getline(iss, token, ','); int output_c = std::stoi(token);
        std::getline(iss, token, ','); int kernel_h = std::stoi(token);
        std::getline(iss, token, ','); int kernel_w = std::stoi(token);
        std::getline(iss, token, ','); int padding_height = std::stoi(token);
        std::getline(iss, token, ','); int padding_width = std::stoi(token);
        std::getline(iss, token, ','); int stride_height = std::stoi(token);
        std::getline(iss, token, ','); int stride_width = std::stoi(token);
        std::getline(iss, token, ','); int dilation_h = std::stoi(token);
        std::getline(iss, token, ','); int dilation_w = std::stoi(token);

        // Initialize descriptors
        cudnnCreateTensorDescriptor(&inputDesc);
        cudnnCreateTensorDescriptor(&outputDesc);
        cudnnCreateFilterDescriptor(&filterDesc);
        cudnnCreateConvolutionDescriptor(&convDesc);

        // Set descriptors
        cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_n, input_c, input_h, input_w);
        cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, output_c, input_c, kernel_h, kernel_w);
        cudnnSetConvolution2dDescriptor(convDesc, padding_height, padding_width, stride_height, stride_width, dilation_h, dilation_w, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);

        // Determine dimensions of the output tensor after convolution
        int n, c, h, w;
        cudnnGetConvolution2dForwardOutputDim(convDesc, inputDesc, filterDesc, &n, &c, &h, &w);
        cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);

        // Use cudnnFindConvolutionForwardAlgorithm to get the best algorithm
        const int requestedAlgoCount = 10;
        int returnedAlgoCount;
        cudnnConvolutionFwdAlgoPerf_t heuristicAlgo[requestedAlgoCount];
        cudnnGetConvolutionForwardAlgorithm_v7(handle, inputDesc, filterDesc, convDesc, outputDesc, requestedAlgoCount, &returnedAlgoCount, heuristicAlgo);

        // Save the results to the dataset
        outfile << line << "," << heuristicAlgo[0].algo << "\n";
        //std::cout << "Algorithm: " << heuristicAlgo[0].algo << ", Time: " << perfResults[0].time << "ms, Memory: " << perfResults[0].memory/1048576 << " MB" << std::endl;
        // Destroy the descriptors
        cudnnDestroyTensorDescriptor(inputDesc);
        cudnnDestroyTensorDescriptor(outputDesc);
        cudnnDestroyFilterDescriptor(filterDesc);
        cudnnDestroyConvolutionDescriptor(convDesc);
    }

    infile.close();
    outfile.close();
    cudnnDestroy(handle);
    return 0;
}
