#include "Neuron.hpp"
#include "util.hpp"
#include "BinaryStream.hpp"

Neuron::Neuron(int input_num, double learning_rate): learning_rate(learning_rate) {
	for (int i = 0; i <= input_num; ++i) { // bias as weight[0]
		weight.push_back((double) rand() / RAND_MAX);
		delta_weight.push_back(0);
	}
}

double Neuron::getWeight(int i) const {
	if (i >= weight.size()) throw "Neuron index out of range";
	return weight[i];
}

double& Neuron::operator[](int i) {
	if (i >= weight.size()) throw "Neuron index out of range";
	return weight[i];
}

double Neuron::operator[](int i) const {
	if (i >= weight.size()) throw "Neuron index out of range";
	return weight[i];
}

void Neuron::cal(const vector<double>& input) {
	if (input.size() != weight.size() - 1) throw "Neuron size error";
	double sum = weight[0]; // bias
	for (int i = 0; i < input.size(); ++i) {
		sum += input[i] * weight[i+1];
	}
	output = sigmoid(sum);
}

void Neuron::calOutputGradient(const double expect_output) {
	gradient = (expect_output - output) * output * (1 - output);
}

void Neuron::calHiddenGradient(const vector<Neuron>& next, int j) {
	double sum = 0;
	for(int i = 0; i < next.size(); ++i) {
		sum += next[i].getGradient() * next[i][j];
	}
	gradient = output * (1 - output) * sum;
}

void Neuron::update(const vector<double>& prev) {
	weight[0] += learning_rate * gradient; // bias
	for (int i = 0; i < prev.size(); ++i) {
		weight[i+1] += learning_rate * gradient * prev[i];
	}
}

double Neuron::calSquareError() {
	double error = 0;
	for (int i = 0; i < weight.size(); ++i) {
		double c = weight[i] - delta_weight[i];
		error += c * c;
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
	for (int i = 0; i < size; i++) {
		n.weight.push_back(bs.readDouble(is));
		n.delta_weight.push_back(0);
	}
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