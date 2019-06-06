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
		weight(w), delta_weight(w), learning_rate(learning_rate) {}
	void setOutput(double a) { output = a; }
	double getOutput() const { return output; }
	double getGradient() const { return gradient; }
	double getWeight(int i) const;
	void cal(const vector<Neuron>& prev);
	void calOutputGradient(const double samp_output);
	void calHiddenGradient(const vector<Neuron>& next, int j);
	void update(const vector<Neuron>& prev);
	double calSquareError();
	friend ostream& operator<<(ostream& os, const Neuron& k);
	friend istream& operator>>(istream& is, Neuron& k);
	void print();
private:
	vector<double> weight; // {weights..., bias}
	double learning_rate;
	vector<double> delta_weight;
	double output;
	double gradient;
};

#endif