#include <iomanip>
#include <cmath>
#include <iostream>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>

#define CSC(call)  									                \
do {											                    \
	cudaError_t res = call;							                \
	if (res != cudaSuccess) {							            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);								                    \
	}										                        \
} while(0)

struct comparator {
    __host__ __device__ bool operator()(double a, double b) {
        return fabs(a) < fabs(b);
    }
};

__global__ void swap(double *mx, int n, int i, int max_i) {
    int id_x = blockDim.x * blockIdx.x + threadIdx.x;
    int offset_x = blockDim.x * gridDim.x;
    for (int x = id_x; x < n; x += offset_x) {
        double temp = mx[i + x * n];
        mx[i + x * n] = mx[max_i + x * n];
        mx[max_i + x * n] = temp;
    }
}

__global__ void update(double *mx, int n, int i) {
    int id_x = blockDim.x * blockIdx.x + threadIdx.x;
    int id_y = blockDim.y * blockIdx.y + threadIdx.y;
    int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;
    for (int x = id_x + i + 1; x < n; x += offset_x) {
        for (int y = id_y + i + 1; y < n; y += offset_y) {
            mx[x + y * n] -= mx[i + y * n] * (mx[x + i * n] / mx[i + i * n]);
        }
    }
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);

    comparator comp;
    int n;
    std::cin >> n;

    double* mx = (double*)malloc(n * n * sizeof(double));
    double x;
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            std::cin >> x;
            mx[i + j * n] = x;
        }
    }

    /*
    for (int i = 0; i < n*n; i++){
        std::cout << mx[i] << " ";
    }
    std::cout << "\n";
    */

    double* dev_mx;
    CSC(cudaMalloc(&dev_mx, n * n * sizeof(double)));
    CSC(cudaMemcpy(dev_mx, mx, n * n * sizeof(double), cudaMemcpyHostToDevice));

    int max_i;
    int swaps_count = 0;
    for(int i = 0; i < n - 1; i++){
        thrust::device_ptr<double> ptr = thrust::device_pointer_cast(dev_mx);
        thrust::device_ptr<double> max_ptr = thrust::max_element(ptr + i * n + i, ptr + i * n + n, comp);
        max_i = max_ptr - ptr - i * n;

        swap<<<128, 128>>>(dev_mx, n, i, max_i);
        if (i != max_i){
            swaps_count++;
        }

        update<<<dim3(16, 16), dim3(32, 32)>>>(dev_mx, n, i);
    }
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(mx, dev_mx, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    CSC(cudaFree(dev_mx));

    double sum_log = 0;
    int count_neg = 0;
    int mult_after = 1.0f;
    for(int i = 0; i < n; i++){
        if(std::abs(mx[i + i * n]) < 1e-7){
            mult_after = 0;
        }
        sum_log += std::log(std::abs(mx[i + i * n]));
        if(mx[i + i * n] < 0){
            count_neg += 1;
        }
    }

    double suma = std::exp(sum_log);
    if(count_neg % 2 != 0){
        suma *= -1.0f;
    }
    if(swaps_count % 2 != 0){
        suma *= -1.0f;
    }
    if(mult_after == 0){
        suma = 0;
    }
    std::cout << std::scientific << std::setprecision(10) << suma << "\n";

    free(mx);
    return 0;
}