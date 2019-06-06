#include "Neuron.hpp"
#include "util.hpp"
#include "BinaryStream.hpp"

Neuron::Neuron(int input_num, double learning_rate): learning_rate(learning_rate) {
	for (int i = 0; i < input_num; ++i) {
		weight.push_back((double) rand() / RAND_MAX);
		delta_weight.push_back(0);
	}
	weight.push_back(1); // bias
	delta_weight.push_back(0);
}

double Neuron::getWeight(int i) const {
	if (i >= weight.size()) throw "Neuron index out of range";
	return weight[i];
}

void Neuron::cal(const vector<Neuron>& prev) {
	if (prev.size() != weight.size()) throw "Neuron size error";
	double sum = 0;
	for (int i = 0; i < weight.size(); ++i) {
		sum += prev[i].getOutput() * weight[i];
	}
	output = sigmoid(sum);
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

double Neuron::calSquareError() {
	double error = 0;
	for (int i = 0; i < weight.size(); ++i) {
		double a = weight[i] - delta_weight[i];
		error += a * a;
		delta_weight[i] = weight[i];
	}
	return error;
}

ostream& operator<<(ostream& os, const Neuron& n) {
	// { int size, double learning_rate, double weight[size] }
	BinaryStream bs;
	// write obj to stream
	bs.writeInt(os, n.weight.size());
	bs.writeDouble(os, n.learning_rate);
	for (int i = 0; i < n.weight.size(); i++)
		bs.writeDouble(os, n.weight[i]);
	return os;
}

istream& operator>>(istream& is, Neuron& n) {
	// { int size, double learning_rate, double weight[size] }
	BinaryStream bs;
	// read obj from stream
	int size = bs.readInt(is);
	n.learning_rate = bs.readDouble(is);
	n.weight.clear();
	for (int i = 0; i < size; i++)
		n.weight.push_back(bs.readDouble(is));
	return is;
}

void Neuron::print() {
	cout << "Neuron:" << endl;
	cout << "learning rate: " << learning_rate << endl;
	cout << "weights:" << endl;
	for (int i = 0; i < weight.size(); ++i)
		cout << weight[i] << " ";
	cout << endl;
}