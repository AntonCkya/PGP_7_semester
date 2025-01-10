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

__global__ void kernel(cudaTextureObject_t tex, uchar4 *out, int w, int h) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
	  int idy = blockDim.y * blockIdx.y + threadIdx.y;
   	int offsetx = blockDim.x * gridDim.x;
	  int offsety = blockDim.y * gridDim.y;
    int x, y;
    for(y = idy; y < h; y += offsety)
        for(x = idx; x < w; x += offsetx) {
            float i[3][3];
            for(int ii = 0; ii <= 2; ii++){
                for(int jj = 0; jj <= 2; jj++){
                    uchar4 p = tex2D<uchar4>(tex, max(min(x+jj-1, w-1), 0), max(min(y+ii-1, h-1), 0));
                    i[ii][jj] = 0.299f * p.x + 0.587f * p.y + 0.114f * p.z;
                }
            }
            /*
            uchar4 W[3][3];
            W[0][0] = tex2D< uchar4 >(tex, max(x-1, 0), max(y-1, 0));
            W[0][1] = tex2D< uchar4 >(tex, x, max(y-1, 0));
            W[0][2] = tex2D< uchar4 >(tex, min(x+1, w-1), max(y-1, 0));
            W[1][0] = tex2D< uchar4 >(tex, max(x-1, 0), y);
            W[1][1] = tex2D< uchar4 >(tex, x, y);
            W[1][2] = tex2D< uchar4 >(tex, min(x+1, w-1), y);
            W[2][0] = tex2D< uchar4 >(tex, max(x-1, 0), min(y+1, h-1));
            W[2][1] = tex2D< uchar4 >(tex, x, min(y+1, h-1));
            W[2][2] = tex2D< uchar4 >(tex, min(x+1, w-1), min(y+1, h-1));

            float i[3][3];
            i[0][0] = 0.299f * W[0][0].x + 0.587f * W[0][0].y + 0.114f * W[0][0].z;
            i[0][1] = 0.299f * W[0][1].x + 0.587f * W[0][1].y + 0.114f * W[0][1].z;
            i[0][2] = 0.299f * W[0][2].x + 0.587f * W[0][2].y + 0.114f * W[0][2].z;
            i[1][0] = 0.299f * W[1][0].x + 0.587f * W[1][0].y + 0.114f * W[1][0].z;
            i[1][1] = 0.299f * W[1][1].x + 0.587f * W[1][1].y + 0.114f * W[1][1].z;
            i[1][2] = 0.299f * W[1][2].x + 0.587f * W[1][2].y + 0.114f * W[1][2].z;
            i[2][0] = 0.299f * W[2][0].x + 0.587f * W[2][0].y + 0.114f * W[2][0].z;
            i[2][1] = 0.299f * W[2][1].x + 0.587f * W[2][1].y + 0.114f * W[2][1].z;
            i[2][2] = 0.299f * W[2][2].x + 0.587f * W[2][1].y + 0.114f * W[2][2].z;
            */

            float gx = i[0][2] + 2.0f*i[1][2] + i[2][2] - i[0][0] - 2.0f*i[1][0] - i[2][0];
            float gy = i[2][0] + 2.0f*i[2][1] + i[2][2] - i[0][0] - 2.0f*i[0][1] - i[0][2];

            float f = sqrtf(gx * gx + gy * gy);
            if (f <= 0.0f){
              f = 0.0f;
            } else if (f >= 255.0f){
              f = 255.0f;
            }

            out[y * w + x] = make_uchar4(f, f, f, 255);
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

    cudaArray *arr;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    CSC(cudaMallocArray(&arr, &ch, w, h));
    CSC(cudaMemcpy2DToArray(arr, 0, 0, data, w * sizeof(uchar4), w * sizeof(uchar4), h, cudaMemcpyHostToDevice));

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = arr;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;

    cudaTextureObject_t tex = 0;
    CSC(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

    uchar4 *dev_out;
	  CSC(cudaMalloc(&dev_out, sizeof(uchar4) * w * h));

    kernel<<< dim3(16, 16), dim3(32, 32) >>>(tex, dev_out, w, h);
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

    CSC(cudaDestroyTextureObject(tex));
    CSC(cudaFreeArray(arr));
    CSC(cudaFree(dev_out));

    fp = fopen(out_name.c_str(), "wb");
    fwrite(&w, sizeof(int), 1, fp);
    fwrite(&h, sizeof(int), 1, fp);
    fwrite(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    free(data);
    return 0;
}
