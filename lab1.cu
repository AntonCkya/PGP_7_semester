#include <stdio.h>

__global__ void kernel(double *arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    while(2 * idx < n) {
        double temp = arr[idx];
        arr[idx] = arr[n - idx - 1];
        arr[n - idx - 1] = temp;
        idx += offset;
    }
}

int main() {
    int n;
    scanf("%d\n", &n);

    double *arr = (double *)malloc(sizeof(double) * n);
    for(int i = 0; i < n; i++)
        scanf("%lf", &arr[i]);
    
    double *dev_arr;
    cudaMalloc(&dev_arr, sizeof(double) * n);
    cudaMemcpy(dev_arr, arr, sizeof(double) * n, cudaMemcpyHostToDevice);

    kernel<<<1024, 1024>>>(dev_arr, n);

    cudaMemcpy(arr, dev_arr, sizeof(double) * n, cudaMemcpyDeviceToHost);
    for(int i = 0; i < n; i++)
        printf("%.10e ", arr[i]);
    printf("\n");

    free(arr);
    cudaFree(dev_arr);
    return 0;
}
