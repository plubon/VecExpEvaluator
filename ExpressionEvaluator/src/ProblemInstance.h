/*
 * ProblemInstance.h
 *
 *  Created on: Apr 26, 2016
 *      Author: piotr
 */
#ifndef PROBLEMINSTANCE_H_
#define PROBLEMINSTANCE_H_

#include <map>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "InfixTree.h"
#include <algorithm>
#include <stack>

using namespace std;

class ProblemInstance {

private:
	map<char, vector<float> > vectors;
	string expression;
	vector<char> variableNames;
	void ExpressionToRpn();
	void CudaMallocError(string, string);

public:
	void AddVector(char, vector<float>);
	vector<float> GetVector(char);
	ProblemInstance(string);
	virtual ~ProblemInstance();
	string GetPostfixExpression();
	string GetInfixTree();
	void CopyToDevice(char*&, float*&, float*&);
	int GetNumOfVariables();
	vector<float> EvaluateCpu();
	int length;
	int ExpLength;
};

#endif /* PROBLEMINSTANCE_H_ */
