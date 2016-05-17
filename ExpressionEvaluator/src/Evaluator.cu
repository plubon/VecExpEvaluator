#include "Evaluator.cuh"

extern __shared__ float shared[];

__global__ void Evaluate(float* vectors, float* results, char* expression, int vectorLen, int numOfVars, int expLen)
{
	unsigned inWholeIdx = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned inBlockIdx = threadIdx.x;
	unsigned alignedNumOfVars = numOfVars;
	if(numOfVars % 4 != 0)
		alignedNumOfVars = alignedNumOfVars + 4 - (numOfVars%4);
	unsigned width = (expLen + 1) / 4;
	unsigned inSubproblemId = threadIdx.x % width;
	unsigned threads = blockDim.x-(blockDim.x%width);
	unsigned subproblemNoInWhole = (blockIdx.x * (blockDim.x/width)) + (inBlockIdx / width);
	unsigned subproblemNoInBlock = inBlockIdx / width;
	unsigned memBlockSize = alignedNumOfVars + width;
	char* exp = (char*)shared;
	float* vals = (float*)&exp[expLen+1];
	unsigned expId =  expLen - ((expLen + 1) / 2) - ((expLen + 1) / 4) + inSubproblemId;
	unsigned varStart = memBlockSize * subproblemNoInBlock;
	unsigned treeStart = varStart + alignedNumOfVars;
	unsigned valId = inSubproblemId + treeStart;
	if(inBlockIdx < expLen)
		exp[inBlockIdx] = expression[inBlockIdx];
	if(subproblemNoInWhole < vectorLen)
	{
		if(inSubproblemId < numOfVars)
			vals[(subproblemNoInBlock*memBlockSize) + inSubproblemId] = vectors[(subproblemNoInWhole*numOfVars) + inSubproblemId];
		if(inSubproblemId + width < numOfVars)
			vals[(subproblemNoInBlock*memBlockSize) + inSubproblemId + width] = vectors[(subproblemNoInWhole*numOfVars) + inSubproblemId + width];
	}
	__syncthreads();
	if(subproblemNoInWhole < vectorLen)
	{
		if(exp[expId] == '+')
			vals[valId]=vals[varStart+exp[expId*2+1]-65]+vals[varStart+exp[expId*2+2]-65];
		else if(exp[expId] == '-')
			vals[valId]=vals[varStart+exp[expId*2+1]-65]-vals[varStart+exp[expId*2+2]-65];
		else if(exp[expId] == '/')
			vals[valId]=vals[varStart+exp[expId*2+1]-65]/vals[varStart+exp[expId*2+2]-65];
		else if(exp[expId] == '*')
			vals[valId]=vals[varStart+exp[expId*2+1]-65]*vals[varStart+exp[expId*2+2]-65];
		else if(exp[expId]>=65 && exp[expId]<=122)
			vals[valId]=vals[varStart+exp[expId]-65];
	}
	__syncthreads();
	unsigned level = 1;
	while(expId!=0)
	{
		expId = (expId -1) / 2;
		if(inSubproblemId%level == 0)
		{
			if(exp[expId] == '+')
				vals[valId]=vals[valId]+vals[valId+level];
			else if(exp[expId] == '-')
				vals[valId]=vals[valId]-vals[valId+level];
			else if(exp[expId] == '/')
				vals[valId]=vals[valId]/vals[valId+level];
			else if(exp[expId] == '*')
				vals[valId]=vals[valId]*vals[valId+level];
			else if(exp[expId]>=65 && exp[expId]<=122)
				vals[valId]=vals[varStart+exp[expId]-65];
			level = level*2;
		}
		 __syncthreads();
	}
	if(inSubproblemId == 0)
		results[subproblemNoInWhole] = vals[valId];
}

__global__ void EvaluateSinglePerBlock(float* vectors, float* results, char* expression, int vectorLen, int numOfVars, int expLen)
{
	unsigned inWholeIdx = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned inBlockIdx = threadIdx.x;
	unsigned alignedNumOfVars = numOfVars;
	if(numOfVars % 4 != 0)
		alignedNumOfVars = alignedNumOfVars + 4 - (numOfVars%4);
	unsigned width = (expLen + 1) / 4;
	unsigned inSubproblemId = threadIdx.x % width;
	unsigned threads = blockDim.x-(blockDim.x%width);
	unsigned subproblemNoInWhole = blockIdx.x;//(blockIdx.x * (blockDim.x/width)) + (inBlockIdx / width);
	unsigned subproblemNoInBlock = inBlockIdx / width;
	unsigned memBlockSize = alignedNumOfVars + width;
	char* exp = (char*)shared;
	float* vals = (float*)&exp[expLen+1];
	unsigned expId =  expLen - ((expLen + 1) / 2) - ((expLen + 1) / 4) + inSubproblemId;
	unsigned varStart = memBlockSize * subproblemNoInBlock;
	unsigned treeStart = varStart + alignedNumOfVars;
	unsigned valId = inSubproblemId + treeStart;
	if(inBlockIdx < expLen)
		exp[inBlockIdx] = expression[inBlockIdx];
	if(subproblemNoInWhole < vectorLen && subproblemNoInBlock == 0)
	{
		if(inSubproblemId < numOfVars)
			vals[(subproblemNoInBlock*memBlockSize) + inSubproblemId] = vectors[(subproblemNoInWhole*numOfVars) + inSubproblemId];
		if(inSubproblemId + width < numOfVars)
			vals[(subproblemNoInBlock*memBlockSize) + inSubproblemId + width] = vectors[(subproblemNoInWhole*numOfVars) + inSubproblemId + width];
	}
	__syncthreads();
	if(subproblemNoInWhole < vectorLen && subproblemNoInBlock == 0)
	{
		if(exp[expId] == '+')
			vals[valId]=vals[varStart+exp[expId*2+1]-65]+vals[varStart+exp[expId*2+2]-65];
		else if(exp[expId] == '-')
			vals[valId]=vals[varStart+exp[expId*2+1]-65]-vals[varStart+exp[expId*2+2]-65];
		else if(exp[expId] == '/')
			vals[valId]=vals[varStart+exp[expId*2+1]-65]/vals[varStart+exp[expId*2+2]-65];
		else if(exp[expId] == '*')
			vals[valId]=vals[varStart+exp[expId*2+1]-65]*vals[varStart+exp[expId*2+2]-65];
		else if(exp[expId]>=65 && exp[expId]<=122)
			vals[valId]=vals[varStart+exp[expId]-65];
	}
	__syncthreads();
	unsigned level = 1;
	while(expId!=0)
	{
		expId = (expId -1) / 2;
		if(inSubproblemId%level == 0 && subproblemNoInBlock == 0)
		{
			if(exp[expId] == '+')
				vals[valId]=vals[valId]+vals[valId+level];
			else if(exp[expId] == '-')
				vals[valId]=vals[valId]-vals[valId+level];
			else if(exp[expId] == '/')
				vals[valId]=vals[valId]/vals[valId+level];
			else if(exp[expId] == '*')
				vals[valId]=vals[valId]*vals[valId+level];
			else if(exp[expId]>=65 && exp[expId]<=122)
				vals[valId]=vals[varStart+exp[expId]-65];
			level = level*2;
		}
		 __syncthreads();
	}
	if(inSubproblemId == 0 && subproblemNoInBlock == 0)
		results[subproblemNoInWhole] = vals[valId];
}
