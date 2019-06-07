#ifndef NN_H
#define NN_H

#include "Neuron.hpp"
#include "util.hpp"
using namespace std;

#define DEFAULT_NN_EPS 1e-3

class NN {
public:
	NN(const vector<int> shape, double eps = DEFAULT_NN_EPS, double learning_rate = DEFAULT_LEARNING_RATE);
	NN(const vector<vector<double> > shape_learning_rate, double eps = DEFAULT_NN_EPS);
	NN(Layers weights, double eps = DEFAULT_NN_EPS, double learning_rate = DEFAULT_LEARNING_RATE);
	vector<double> getOutput(int layer);
	void forward(const vector<double>& input);
	void backProp(const vector<double>& expect_output);
	vector<double> getResult(const vector<double>& input);
	double calStandardError();
	friend ostream& operator<<(ostream& os, const NN& nn);
	friend istream& operator>>(istream& is, NN& nn);
	void print();
	double eps;
private:
	vector<vector<Neuron> > net;
	vector<double> last_input;
};

#endif