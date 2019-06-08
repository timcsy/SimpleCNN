#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include "Kernel.hpp"
#include "util.hpp"
using namespace std;

class Convolution {
public:
	Convolution() {}
	Convolution(Layers kernels, int strides = 1, int padding = 0, bool relu = true, int ph = 2, int pw = 2):
		strides(strides), padding(padding), relu(relu), pooling_height(ph), pooling_width(pw) {}
	Convolution(int kn, int kh = 4, int kw = 4, int strides = 1, int padding = 0, bool relu = true, int ph = 2, int pw = 2);
	vector<Kernel> getKernels() { return kernels; }
	void feed(Layers map) { input_map = map; }
	Layers conv();
	Layers max_pooling();
	vector<double> flatten();
	friend ostream& operator<<(ostream& os, const Convolution& c);
	friend istream& operator>>(istream& is, Convolution& c);
	void print();
	void print_input();
	void print_kernels();
	void print_conv();
	void print_pooling();
	void print_flatten();
private:
	vector<Kernel> kernels;
	int strides;
	int padding;
	bool relu;
	int pooling_height;
	int pooling_width;
	Layers input_map;
	Layers conv_map;
	Layers pooling_map;
	vector<double> output_vector;
};

#endif