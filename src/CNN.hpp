#ifndef CNN_H
#define CNN_H

#include "Convolution.hpp"
#include "NN.hpp"

class CNN {
public:
	CNN() {}
	CNN(Layers config);
	vector<double> getResult(const vector<double>& input);
	double train(Records train_data, bool show = false);
	double test(Records test_data);
	friend ostream& operator<<(ostream& os, const CNN& cnn);
	friend istream& operator>>(istream& is, CNN& cnn);
	void print();
private:
	vector<Convolution> conv_layers;
	NN nn;
};

#endif