/*
 * InfixTree.cpp
 *
 *  Created on: Apr 26, 2016
 *      Author: piotr
 */

#include "InfixTree.h"

InfixTree::InfixTree(string input)
{
	// TODO Auto-generated constructor stub
	int size = input.size();
	Node* stack[size] ;
	int stackPos = -1;
	for(int i = 0; i < size; i++)
	{
		if(input[i] >= 'A' && input[i] <= 'z')
		{
			Node *temp = new Node;
			temp->left = temp->right = NULL;
			temp->data = input[i];
			stackPos++;
			stack[stackPos] = temp;
		}
		else
		{
			Node *temp = new Node;
			temp->left = temp->right = NULL;
			temp->data = input[i];
			temp->right = stack[stackPos];
			stackPos--;
			temp->left = stack[stackPos];
			stack[stackPos] = temp;
		}
	}
	this->root = stack[0];
	this->Height = GetHeight(this->root);
}

int InfixTree::GetHeight(Node* tree)
{
	if (tree == NULL)
	{
		return 0;
	}

	int lefth = GetHeight(tree->left);
	int righth = GetHeight(tree->right);

	if(lefth > righth)
	{
		return lefth + 1;
	}
	else
	{
		return righth + 1;
	}
}

InfixTree::~InfixTree() {
	delete this->root;
}

string InfixTree::ToString()
{
	int len = floor(pow(2.0f, this->Height) - 0.5);
	Node* queue[len];
	int queueWrite = 1;
	string order;
	queue[0] = this->root;
	int queueRead = 0;
	while(queueRead < len)
	{
		if(queue[queueRead] == NULL)
			order += '0';
		else
			order += (queue[queueRead]->data);
		if(queueWrite < len)
		{
			if(queue[queueRead] == NULL)
			{
				queue[queueWrite] = NULL;
				queueWrite++;
				queue[queueWrite] = NULL;
				queueWrite++;
			}
			else
			{
				queue[queueWrite] = queue[queueRead]->left;
				queueWrite++;
				queue[queueWrite] = queue[queueRead]->right;
				queueWrite++;
			}
		}
		queueRead++;
	}
	return order;
}
