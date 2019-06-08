#ifndef NN_H
#define NN_H

#include "Neuron.hpp"
#include "Records.hpp"
#include "util.hpp" // type Layers
using namespace std;

#define DEFAULT_NN_EPS 1e-3
#define DEFAULT_NN_N 0

class NN {
public:
	NN() {}
	NN(const vector<int> shape, double eps = DEFAULT_NN_EPS, int N = DEFAULT_NN_N, double learning_rate = DEFAULT_LEARNING_RATE);
	NN(Config shape_learning_rate, double eps = DEFAULT_NN_EPS, int N = DEFAULT_NN_N);
	NN(Layers weights, double eps = DEFAULT_NN_EPS, int N = DEFAULT_NN_N, double learning_rate = DEFAULT_LEARNING_RATE);
	vector<double> getOutput(int layer);
	void forward(const vector<double>& input);
	void backProp(const vector<double>& expect_output);
	vector<double> getResult(const vector<double>& input);
	double calStandardError();
	double sample_error(const Records& data);
	double train(Records train_data, bool show = false);
	double test(Records test_data);
	friend ostream& operator<<(ostream& os, const NN& nn);
	friend istream& operator>>(istream& is, NN& nn);
	void print();
private:
	vector<vector<Neuron> > net;
	vector<double> last_input;
	double eps; // eps == 0: just depend on N
	int N; // the most iterations, N == 0: just depend on eps
};

#endif