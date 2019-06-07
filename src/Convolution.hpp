#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include "Kernel.hpp"
#include "util.hpp"
using namespace std;

class Convolution {
public:
	Convolution(Layers kernels, int strides = 1, int padding = 0):
		strides(strides), padding(padding) {}
	Convolution(int kn, int kh = 4, int kw = 4, int strides = 1, int padding = 0);
	vector<Kernel> getKernels() { return kernels; }
	void feed(Layers map) { input_map = map; }
	Layers conv(bool relu = true);
	Layers max_pooling(int m, int n);
	vector<double> flatten();
	void print_input();
	void print_kernels();
	void print_conv();
	void print_pooling();
	void print_flatten();
private:
	Layers input_map;
	vector<Kernel> kernels;
	Layers conv_map;
	Layers pooling_map;
	vector<double> output_vector;
	int strides;
	int padding;
};

#endif