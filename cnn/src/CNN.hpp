#ifndef CNN_H
#define CNN_H

#include "Convolution.hpp"
#include "NN.hpp"

class CNN {
public:
	CNN() {}
	CNN(Layers config, vector<string> labels);
	void setConvolution(Config config);
	void setNN(Config config);
	vector<double> feed_conv(Layers& input);
	string getResult(vector<double> input);
	Records conv(Records& train_data, bool show = false);
	double train_nn(Records& train_data, bool show = false);
	double train(Records& train_data, bool show = false);
	double test_nn(Records& test_data, bool show = false);
	double test(Records& test_data, bool show = false);
	friend ostream& operator<<(ostream& os, const CNN& cnn);
	friend istream& operator>>(istream& is, CNN& cnn);
	void print();
private:
	vector<Convolution> conv_layers;
	vector<string> labels;
	NN nn;
	int map_height;
	int map_width;
	int map_depth;
};

#endif