#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include "Kernel.hpp"
using namespace std;

typedef vector<vector<vector<double> > > Layers;

class Convolution {
public:
	Convolution(Layers map, Layers kernels, int strides = 1, int padding = 0):
		input_map(map), strides(strides), padding(padding) {}
	Convolution(Layers map, int kn, int kh = 4, int kw = 4, int strides = 1, int padding = 0);
	vector<Kernel> getKernels() { return kernels; }
	Layers conv(bool relu = true);
	Layers max_pooling(int m, int n);
	vector<double> flatten();
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