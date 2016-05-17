/*
 * Evaluator.cuh
 *
 *  Created on: May 17, 2016
 *      Author: piotr
 */

#ifndef EVALUATOR_CUH_
#define EVALUATOR_CUH_
__global__ void Evaluate(float*, float*, char*, int, int, int);
__global__ void EvaluateSinglePerBlock(float*, float*, char*, int, int, int);
#endif /* EVALUATOR_CUH_ */
