/*
 * InfixTree.h
 *
 *  Created on: Apr 26, 2016
 *      Author: piotr
 */

#ifndef INFIXTREE_H_
#define INFIXTREE_H_
#include <string>
#include <math.h>
#include <iostream>

using namespace std;

class InfixTree {
private:
	class Node
	{
		public:
			char data;
			Node *right, *left;
			virtual ~Node()
			{
				//delete left;
				//delete right;
			};
	};
	Node *root;
	int GetHeight(Node*);
public:
	InfixTree(string);
	virtual ~InfixTree();
	string ToString();
	int Height;
};

#endif /* INFIXTREE_H_ */
