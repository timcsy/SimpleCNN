#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <iostream>
using namespace std;

#define DEFAULT_LEARNING_RATE 0.5

class Neuron {
public:
	Neuron(): learning_rate(DEFAULT_LEARNING_RATE) {}
	Neuron(int input_num, double learning_rate = DEFAULT_LEARNING_RATE);
	Neuron(vector<double> w, double learning_rate = DEFAULT_LEARNING_RATE):
		weight(w), delta_weight(vector<double>(w.size(), 0)), learning_rate(learning_rate) {}
	double getOutput() const { return output; }
	double getGradient() const { return gradient; }
	double getWeight(int i) const;
	double& operator[](int i);
	double operator[](int i) const;
	int size() { return weight.size(); }
	void cal(const vector<double>& input);
	void calOutputGradient(const double expect_output);
	void calHiddenGradient(const vector<Neuron>& next, int j);
	void update(const vector<double>& prev);
	double calSquareError();
	friend ostream& operator<<(ostream& os, const Neuron& n);
	friend istream& operator>>(istream& is, Neuron& n);
	void print();
private:
	vector<double> weight; // [ weight_of_bias, weights... ], where bias is +1
	double learning_rate;
	vector<double> delta_weight;
	double output;
	double gradient;
};

#endif