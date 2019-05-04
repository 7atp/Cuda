#include <stdlib.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h" 
#include <omp.h>

void fillVector(int * v, size_t n);
void addVector(int * a, int *b, int *c, size_t n);
void printVector(int * v, size_t n);
void serial();
void cudacode();
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size,int n);

int main()
{
	const int n = 2;
	cudaEvent_t start; cudaEventCreate(&start);
	cudaEvent_t stop; cudaEventCreate(&stop);
	const int vectorSize = n*1024;
	int a[vectorSize], b[vectorSize], c[vectorSize];

	fillVector(a, vectorSize);
	fillVector(b, vectorSize);
	float msecTotal = 0.0f;
	for (int i = 0; i < 50; i++) {
		cudaEventRecord(start, NULL);
		
		addWithCuda(c, a, b, vectorSize,n);
		cudaEventRecord(stop, NULL);
		cudaEventSynchronize(stop);
		float msecTotal1 = 0.0f;
		cudaEventElapsedTime(&msecTotal1, start, stop);
		msecTotal += msecTotal1;
	}
	printf("this is the mean elapsed time %f", msecTotal/50.0);
	/*double starttime, elapsedtime;

	starttime = omp_get_wtime();

	addVector(a, b, c, vectorSize);


	elapsedtime = omp_get_wtime() - starttime;
	printf("this is the mean elapsed time for serial %f", elapsedtime ); */


	printVector(c, vectorSize);

	return EXIT_SUCCESS;
}

__global__ void addKernel(int *c, const int *a, const int *b, const int *n) {
	int i = threadIdx.x;
	for (int j = 0; j < n[0]; j++) {
		c[i+j] = a[i+j] + b[i+j];
	}
}
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size,int n){

	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	int *dev_n = 0;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0); 
	if (cudaStatus != cudaSuccess) {
		printf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}

	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!"); }
	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!"); }
	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!"); }

	cudaStatus = cudaMalloc((void**)&dev_n,  sizeof(int));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!"); }

	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice); 
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); }
	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); }

	cudaStatus = cudaMemcpy(dev_n, &n,sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); }

	addKernel <<<1, 1024 >>> (dev_c, dev_a, dev_b,dev_n);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	}
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); }
	cudaFree(dev_c); cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_n);

	return cudaStatus;



}



// Fills a vector with data
void fillVector(int * v, size_t n) {
	int i;
	for (i = 0; i < n; i++) {
		v[i] = i;
	}
}

// Adds two vectors
void addVector(int * a, int *b, int *c, size_t n) {
	int i;
	for (i = 0; i < n; i++) {
		c[i] = a[i] + b[i];
	}
}

// Prints a vector to the stdout.
void printVector(int * v, size_t n) {
	int i;
	printf("[-] Vector elements: ");
	for (i = 0; i < n; i++) {
		printf("%d, ", v[i]);
	}
	printf("\b\b  \n");
}
