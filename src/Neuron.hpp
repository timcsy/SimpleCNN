#ifndef NEURON_H
#define NEURON_H

#include <cstdio>
#include <vector>
#include <iostream>
#include <cmath>
using namespace std;

#define learning_rate 0.5

double sigmoid(double a) {
	return (1 / (1 + exp(-a)));
}

class Neuron {
public:
	Neuron(const int a): size(a + 1) {
		for (int i = 0; i < a; ++i) {
			weight.push_back(rand() / (double)RAND_MAX);
			delta_weight.push_back(0);
			#ifdef DEBUG
			cout << weight[i] << " ";
			#endif
		}
		weight.push_back(1); // bias
		delta_weight.push_back(0);
		#ifdef DEBUG
		cout << weight[size - 1] << endl;
		#endif	
	}
	Neuron(const int a, const vector<double>& w): size(a + 1) {
		for (int i = 0; i <= a; ++i) {
			weight.push_back(w[i]);
			delta_weight.push_back(w[i]);
			#ifdef DEBUG
			cout << weight[i] << " ";
			#endif	
		}
		#ifdef DEBUG
		cout << endl;
		#endif	
	}
	void setOutput(double a) { output = a; }
	double getOutput() const { return output; }
	double getGradient() const { return gradient; }
	double getWeight(int i) const { assert(i < size); return weight[i]; }
	void cal(const vector<Neuron>& prev);
	void calOutputGradient(const double samp_output);
	void calHiddenGradient(const vector<Neuron>& next, int j);
	void update(const vector<Neuron>& prev);
	double calSquareError(){
		double error = 0;
		for(int i=0;i<weight.size();++i){
			double a = weight[i] - delta_weight[i];
			error += a*a;
			delta_weight[i] = weight[i];
		}
		return error;
	}
	void printWeight(){
		cout<<"weight is ";
		for (int i = 0; i < weight.size(); ++i)
			cout << weight[i] << " ";
		cout << endl;
	}
private:
	vector<double> weight;
	vector<double> delta_weight;
	double output;
	int size;
	double gradient;
};

void Neuron::cal(const vector<Neuron>& prev) {
	assert(prev.size() == size);
	double sum = 0;
	for (int i = 0; i < size; ++i) {
		#ifdef DEBUG
		printf("j = %d\n", i);
		cout << "prev output = " << prev[i].getOutput() << " weight =" << weight[i] << endl;
		#endif	
		sum += prev[i].getOutput() * weight[i];
	}
	output = sigmoid(sum);
	#ifdef DEBUG
	printf("output = %lf\n", output);
	#endif	
}
void Neuron::calOutputGradient(const double samp_output) {
	gradient = (samp_output - output) * output * (1 - output);
}

void Neuron::calHiddenGradient(const vector<Neuron>& next, int j) {
	double t = 0;
	for(int i = 0; i < next.size() - 1; ++i) {
		t += next[i].getGradient() * next[i].getWeight(j);
	}
	gradient = output * (1 - output) * t;
}

void Neuron::update(const vector<Neuron>& prev) {
	for (int i = 0; i < weight.size(); ++i) {
		weight[i] += learning_rate * gradient * prev[i].getOutput();
	}
}

#endif