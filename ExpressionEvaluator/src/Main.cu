/*
 ============================================================================
 Name        : Main.cu
 Author      : Piotr Luboń
 Version     :
 Copyright   :
 Description :
 ============================================================================
 */
#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <unistd.h>
#include "ProblemInstance.h"
#include "Evaluator.cuh"
#include <cfloat>
#include <sys/time.h>
#include <ctime>

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
#define MAX_BLOCK_SIZE 512
typedef long long int64; typedef unsigned long long uint64;


int main(int argc, char * argv[])
{
	cudaEvent_t start, stop, startCopy, endCopy, startSingle, endSingle;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&startCopy);
	cudaEventCreate(&endCopy);
	cudaEventCreate(&startSingle);
	cudaEventCreate(&endSingle);
	char cwd[1024];
	if (getcwd(cwd, sizeof(cwd)) != NULL)
	       fprintf(stdout, "Current working dir: %s\n", cwd);
	string filename="testfile";
	ProblemInstance problem(filename);
	float finalResult[problem.length];
	char* devExpression = NULL;
	float* devVectors = NULL;
	float* devResult = NULL;
	cudaEventRecord(startCopy, 0);
	problem.CopyToDevice(devExpression, devVectors, devResult);
	cudaEventRecord(endCopy, 0);
	cudaEventSynchronize(endCopy);
	float elapsedCopy;
	cudaEventElapsedTime(&elapsedCopy, startCopy, endCopy);
	int explen = problem.ExpLength;
	int width = (explen+1)/4;
	int threads = MAX_BLOCK_SIZE-(MAX_BLOCK_SIZE%width);
	int blocks = ((problem.length * width) / threads);
	if(problem.length*width%threads!=0)
		blocks++;
	int aligned = problem.GetNumOfVariables();
	if(aligned%4!=0)
			aligned = problem.GetNumOfVariables() + 4 - (problem.GetNumOfVariables()%4);
	size_t sharedPerBlock = (threads + (aligned * (MAX_BLOCK_SIZE/width)))  * sizeof(float)+ ((explen+1) * sizeof(char));
	cout<<"Problem length: "<<problem.length<<endl;
	cout<<"Array tree length: "<<explen<<endl;
	cout<<"Threads per subproblem: "<<width<<endl;
	cout<<"Threads per block: "<<threads<<endl;
	cout<<"Blocks: "<<blocks<<endl;
	cout<<"Bytes of shared memory per block: "<<sharedPerBlock<<endl;
	cudaEventRecord(start, 0);
	Evaluate<<<blocks, MAX_BLOCK_SIZE, sharedPerBlock>>>(devVectors, devResult, devExpression, problem.length, problem.GetNumOfVariables(), explen);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaMemcpy(finalResult, devResult, problem.length*sizeof(float), cudaMemcpyDeviceToHost);
	struct timeval t5, t6;
	gettimeofday(&t5, NULL);
	vector<float> cpuResults = problem.EvaluateCpu();
	gettimeofday(&t6, NULL);
	uint64 diff = ((t6.tv_sec - t5.tv_sec) * 1000) +(t6.tv_usec/1000 - t5.tv_usec/1000);
	bool ok = true;
	for(int i = 0; i < problem.length; i++)
	{
		if(!(fabs(cpuResults[i] - finalResult[i]) < FLT_EPSILON))
		{
			cout<<i<<" "<<cpuResults[i]<<" : "<<finalResult[i]<<endl;
			ok = false;
			break;
		}
	}
	if(ok)
		cout<<"Results ok"<<endl;
	else
		cout<<"Results not ok"<<endl;
	memset(finalResult, 0, problem.length * sizeof(float));
	cudaFree(devExpression);
	cudaFree(devVectors);
	cudaFree(devResult);
	cout<<"Copying time: "<<elapsedCopy<<endl;
	cout<<"Calculation time: "<<elapsedTime<<endl;
	cout<<"Cpu calculation time:"<<diff<<endl;
	cudaFree(devExpression);
	cudaFree(devVectors);
	cudaFree(devResult);
}



