#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define CSC(call)  									                \
do {											                    \
	cudaError_t res = call;							                \
	if (res != cudaSuccess) {							            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);								                    \
	}										                        \
} while(0)

__constant__ float3 avg_norm_const[32];
__constant__ int nc_const;

__device__ int calculate_class(uchar4 p){
    float val;
    float max_val = -1.0, max_idx = 0;
    for (int i = 0; i < nc_const; i++){
        val = p.x * avg_norm_const[i].x + p.y * avg_norm_const[i].y + p.z * avg_norm_const[i].z;
        if (val > max_val){
            max_val = val;
            max_idx = i;
        }
    }
    return max_idx;
}

__global__ void kernel(uchar4 *data, int w, int h) {
    int id_x = blockDim.x * blockIdx.x + threadIdx.x;
    int offset_x = blockDim.x * gridDim.x;
    for (int x = id_x; x < w * h; x += offset_x) {
        data[x].w = calculate_class(data[x]);
    }
}

int main() {
    int w, h;
    std::string in_name, out_name;
    std::cin >> in_name;
    std::cin >> out_name;
   	FILE *fp = fopen(in_name.c_str(), "rb");
    fread(&w, sizeof(int), 1, fp);
    fread(&h, sizeof(int), 1, fp);
    uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
    fread(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    int nc;
    std::cin >> nc;
    CSC(cudaMemcpyToSymbol(nc_const, &nc, sizeof(int), 0, cudaMemcpyHostToDevice));

    float3 avg_norm[nc];
    for (int i = 0; i < nc; i++){
        int np;
        std::cin >> np;
        float avg_j_x = 0.0f;
        float avg_j_y = 0.0f;
        float avg_j_z = 0.0f;
        for (int j = 0; j < np; j++){
            int x, y;
            std::cin >> x >> y;
            uchar4 p = data[y * w + x];
            avg_j_x += p.x;
            avg_j_y += p.y;
            avg_j_z += p.z;
        }
        avg_j_x /= np;
        avg_j_y /= np;
        avg_j_z /= np;
        float avg_j_norm = sqrtf(avg_j_x*avg_j_x + avg_j_y*avg_j_y + avg_j_z*avg_j_z);
        avg_norm[i] = make_float3(avg_j_x / avg_j_norm, avg_j_y / avg_j_norm, avg_j_z / avg_j_norm);
    }
    CSC(cudaMemcpyToSymbol(avg_norm_const, avg_norm, sizeof(float3) * nc, 0, cudaMemcpyHostToDevice));

    uchar4 *dev_out;
    CSC(cudaMalloc(&dev_out, sizeof(uchar4) * w * h));
    CSC(cudaMemcpy(dev_out, data, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));

    kernel<<<1024, 1024>>>(dev_out, w, h);
    
    CSC(cudaDeviceSynchronize());
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));
    CSC(cudaFree(dev_out));

    fp = fopen(out_name.c_str(), "wb");
    fwrite(&w, sizeof(int), 1, fp);
    fwrite(&h, sizeof(int), 1, fp);
    fwrite(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    free(data);
    return 0;
}