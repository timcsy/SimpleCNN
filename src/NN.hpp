#ifndef NN_H
#define NN_H

#include "Neuron.hpp"
#include "util.hpp"
using namespace std;

class NN {
public:
	NN(const vector<int> shape);
	NN(const vector<int> shape, Layers weights);
	void forward(const vector<double>& input);
	void backProp(const vector<double>& samp_output);
	void getResult(const vector<double>& input, vector<double>& ans);
	double calStandardError();
	friend ostream& operator<<(ostream& os, const NN& nn);
	friend istream& operator>>(istream& is, NN& nn);
	void print();
private:
	vector<vector<Neuron> > net;
};

#endif