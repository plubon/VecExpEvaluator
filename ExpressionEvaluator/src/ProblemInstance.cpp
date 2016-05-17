/*
 * ProblemInstance.cpp
 *
 *  Created on: Apr 26, 2016
 *      Author: piotr
 */

#include "ProblemInstance.h"

ProblemInstance::ProblemInstance(string filename)
{
	ifstream file(filename.c_str());
	string line;
	int j = 0;
	this->length = 0;
	while(getline(file, line))
	{
		if(j == 0)
		{
			this->expression = line;
		}
		else if (j == 1)
		{
			istringstream ss(line);
			vector<string> tokens;
			string token;
			while(std::getline(ss, token, ','))
				tokens.push_back(token);
			for(int i = 0; i < tokens.size(); i++)
			{
				this->variableNames.push_back(tokens[i][0]);
				vector<float> temp;
				this->vectors.insert(pair<char,vector<float> >(tokens[i][0], temp));
			}
			for(int i=0; i<this->variableNames.size(); i++)
			{
				char newVal = i + 65;
				replace(this->expression.begin(), this->expression.end(), this->variableNames[i], newVal);
				vector<float> temp = this->vectors[variableNames[i]];
				this->vectors.erase(variableNames[i]);
				AddVector(newVal, temp);
				variableNames[i] = newVal;
			}
		}
		else
		{
			istringstream ss(line);
			string token;
			int i =0;
			while(std::getline(ss, token, ','))
			{
				this->vectors[this->variableNames[i]].push_back(atof(token.c_str()));
				i++;
			}
		}
		j++;
	}
	this->length = j - 2;
	this->ExpressionToRpn();
}

ProblemInstance::~ProblemInstance() {
	// TODO Auto-generated destructor stub
}

void ProblemInstance::AddVector(char VarName, vector<float> values)
{
	this->vectors.insert(pair<char,vector<float> >(VarName,values));
}

vector<float> ProblemInstance::GetVector(char var)
{
	return this->vectors[var];
}

string ProblemInstance::GetPostfixExpression()
{
	return this->expression;
}

string ProblemInstance::GetInfixTree()
{
	InfixTree tree(this->expression);
	string ret = tree.ToString();
	this->ExpLength = ret.size();
	return ret;
}

void ProblemInstance::CudaMallocError(string op,string memory)
{
	fprintf(stderr, "Failed to %s for %s!\n",op.c_str(), memory.c_str());
	exit(EXIT_FAILURE);
}

void ProblemInstance::CopyToDevice(char*& expression, float*& vectors, float*& result)
{
	string tree = this->GetInfixTree();
	cudaError_t err = cudaSuccess;
	err = cudaMalloc((void **)&expression, tree.size() * sizeof(char));
	if (err != cudaSuccess)
		CudaMallocError("allocate","expression");
	err = cudaMalloc((void**)&vectors, this->variableNames.size()*this->length*sizeof(float));
	if (err != cudaSuccess)
			CudaMallocError("allocate","vectors");
	err = cudaMalloc((void**)&result, this->length*sizeof(float));
	if (err != cudaSuccess)
				CudaMallocError("allocate","result");
	err = cudaMemcpy(expression, tree.c_str(), tree.size()*sizeof(char), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
			CudaMallocError("copy","expression");
	int len = this->variableNames.size() * this->length;
	float* data = (float*) malloc(len * sizeof(float));
	for(int i = 0; i < this->length; i++)
	{
		for(int j = 0; j < this->variableNames.size(); j++)
		{
			data[j+(i*this->variableNames.size())] = this->vectors[this->variableNames[j]][i];
		}
	}
	err = cudaMemcpy(vectors, data, len * sizeof(float), cudaMemcpyHostToDevice);
	free(data);
	if (err != cudaSuccess)
				CudaMallocError("copy","vectors");
}

int ProblemInstance::GetNumOfVariables()
{
	return this->variableNames.size();
}

vector<float> ProblemInstance::EvaluateCpu()
{
	vector<float> results;
	for(int i=0;i<this->length;i++)
	{
		stack<float> stos;
		for(int j=0; j<this->expression.size(); j++)
		{
			if(expression[j]>=65 && expression[j]<=122)
				stos.push(vectors[expression[j]][i]);
			else
			{
				float first = stos.top();
				stos.pop();
				float second = stos.top();
				stos.pop();
				if(expression[j] == '+')
					stos.push(first+second);
				else if(expression[j] == '-')
					stos.push(second-first);
				else if(expression[j]== '/')
					stos.push(second/first);
				else if(expression[j] == '*')
					stos.push(first*second);
			}
		}
		results.push_back(stos.top());
	}
	return results;
}

void ProblemInstance::ExpressionToRpn()
{
	int size = this->expression.size();
	string output;
	char stack[size] ;
	int stackPos = -1;
	unsigned pos = 0;
	while(pos < size)
	{
		if(this->expression[pos] >= 'A' && this->expression[pos] <= 'z')
		{
			output += (this->expression[pos]);
		}
		if(this->expression[pos] == '/' || this->expression[pos] == '*')
		{
			while(stackPos >=0 && (stack[stackPos] == '/' || stack[stackPos] == '*'))
			{
				output += (stack[stackPos]);
				stackPos--;
			}
			stackPos++;
			stack[stackPos] = this->expression[pos];
		}
		if(this->expression[pos] == '+' || this->expression[pos] == '-')
		{
			while(stackPos >=0 && (stack[stackPos] == '/' || stack[stackPos] == '*' || stack[stackPos] == '+' || stack[stackPos] == '-'))
			{
				output += (stack[stackPos]);
				stackPos--;
			}
			stackPos++;
			stack[stackPos] = this->expression[pos];
		}
		if(this->expression[pos] == '(')
		{
			stackPos++;
			stack[stackPos] = this->expression[pos];
		}
		if(this->expression[pos] == ')')
		{
			while(stackPos >=0 && stack[stackPos] != '(')
			{
				output += (stack[stackPos]);
				stackPos--;
			}
			stackPos--;
		}
		pos++;
	}
	while(stackPos >= 0)
	{
		output += (stack[stackPos]);
		stackPos--;
	}
	this->expression = output;
}
