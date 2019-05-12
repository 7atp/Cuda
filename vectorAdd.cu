#include <stdlib.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h" 
#include <omp.h>

//set param here
#define block_numbers  2
#define thread_numbers  64
#define jobs_number  1000



void fillVector(int * v, size_t n);
void addVector(int * a, int *b, int *c, size_t n);
void addOMP(int * a, int *b, int *c, int n);

void printVector(int * v, size_t n);
void serial();
void cudacode();
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

int main()
{
	
	cudaEvent_t start; cudaEventCreate(&start);
	cudaEvent_t stop; cudaEventCreate(&stop);
	const int vectorSize = jobs_number;
	int a[vectorSize], b[vectorSize], c[vectorSize];


	/////////////THIS is CUDA part

	fillVector(a, vectorSize);
	fillVector(b, vectorSize);
	float msecTotal = 0.0f;

		cudaEventRecord(start, NULL);
		
		addWithCuda(c, a, b, vectorSize);
		cudaEventRecord(stop, NULL);
		cudaEventSynchronize(stop);
		float msecTotal1 = 0.0f;
		cudaEventElapsedTime(&msecTotal1, start, stop);
		msecTotal += msecTotal1;
	
	
	printf("this is the mean elapsed time for cuda %f \n", msecTotal/1.0);
	
	///////////////////////




	//THIS is OPENMP part
	/*fillVector(a, vectorSize);
	fillVector(b, vectorSize);
	
	double starttime, elapsedtime;
	starttime = omp_get_wtime();
	addOMP(a, b, c, (int)vectorSize);
	elapsedtime = omp_get_wtime() - starttime;
	printf("this is the mean elapsed time for OMP %f", elapsedtime); */





	// THIS is SERIAL part
	/*
	double starttime, elapsedtime;

	starttime = omp_get_wtime();

	addVector(a, b, c, vectorSize);


	elapsedtime = omp_get_wtime() - starttime;
	printf("this is the mean elapsed time for serial %f", elapsedtime ); */


	//printVector(c, vectorSize);

	return EXIT_SUCCESS;
}

void addOMP(int * a, int *b, int *c, int size) {


	int chunck = (int)size/ 8;
#pragma omp parallel for num_threads(8)	private(j,k) 
	for (int i = 0; i <= chunck; i++) {
		int j = i * 8;
		for (int k = 0; k < 8; k++) {
			if (j + k >= size)break;
			c[j+k] = a[j + k] + b[j + k];
		}
	}

}





__global__ void addKernel(int *c, const int *a, const int *b, const int *n,long int* report) {

	long int smid;  
	asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));

	int i = blockIdx.x*blockDim.x+threadIdx.x;
	report[i * 5] = i;
	report[i * 5 + 1] = smid;
	report[i * 5 + 2] = blockIdx.x;
	report[i * 5 + 3] = i%32;
	report[i * 5 + 4] = threadIdx.x;


	int k = i * n[1];
	for (int j = 0; j < n[1]; j++) {
		if (k + j >= n[0])break;
		report[(k+j) * 5] = k+j;
		report[(k + j) * 5 + 1] = smid;
		report[(k + j) * 5 + 2] = blockIdx.x;
		report[(k + j) * 5 + 3] = i % 32;
		report[(k + j) * 5 + 4] = threadIdx.x;

		c[k+j] = a[k+j] + b[k+j];
	}
}
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size){
	 
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	int *dev_n = 0;
	int batch = ((int)size )/ (block_numbers*thread_numbers);
	int temp[2] = {(int)size,batch+1};
	long int *dev_report = 0;
	//printf("sdfafadfv %d ----- %d",temp[0],temp[1]);

	long int * report = (long int *)malloc(5*size * sizeof(long int));

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

	cudaStatus = cudaMalloc((void**)&dev_report, 5*size * sizeof(long int));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!"); }

	cudaStatus = cudaMalloc((void**)&dev_n,  2*sizeof(int));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!"); }

	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice); 
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); }
	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); }

	cudaStatus = cudaMemcpy(dev_n, &temp,2*sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); }

	addKernel <<<block_numbers, thread_numbers >>> (dev_c, dev_a, dev_b,dev_n,dev_report);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	}
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); }
	cudaStatus = cudaMemcpy(report, dev_report, 5*size * sizeof(long int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); }

	for (int i = 0; i < 550; i++) {

	}
	for (int i = 0; i < size; i++) {
		printf("c[%d] == %d ",i,c[i]);
		printf(" ===> Worker %d , SMID %d , BlockId %d , Warp %d , Thread %d \n", report[i * 5], report[i * 5 + 1], report[i * 5 + 2], report[i * 5 + 3], report[i * 5 + 4]);


	}
	cudaFree(dev_c); cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_n); cudaFree(dev_report);

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
