#include "NN.hpp"
#include "BinaryStream.hpp"
using namespace std;

NN::NN(const vector<int> shape) {
	// input layer
	vector<Neuron> layer;
	for (int j = 0; j <= shape[0]; ++j)
		layer.push_back(Neuron(0));
	net.push_back(layer);
	// hidden and output layer
	for (int i = 1; i < shape.size(); ++i) {
		vector<Neuron> layer;
		for (int j = 0; j <= shape[i]; ++j)
			layer.push_back(Neuron(shape[i-1])); // number of prev layer neurons
		net.push_back(layer);
	}
}

NN::NN(const vector<int> shape, Layers weights) {
	// input layer
	vector<Neuron> layer;
	for (int j = 0; j <= shape[0]; ++j) // input + 1 neurons
		layer.push_back(Neuron(0));
	net.push_back(layer);
	// hidden and output layer
	for (int i = 1; i < shape.size(); ++i) {
		vector<Neuron> layer;
		for (int j = 0; j < shape[i]; ++j)
			layer.push_back(Neuron(weights[i-1][j]));
		layer.push_back(Neuron(shape[i-1])); // include bias
		net.push_back(layer);
	}
}

void NN::forward(const vector<double>& input) {
	if (input.size() != net[0].size() - 1) throw "NN size error";
	// assign input value
	for (int i = 0; i < input.size(); ++i)
		net[0][i].setOutput(input[i]);
	net[0][input.size()].setOutput(1); // bias

	// feed dorward
	for (int i = 1; i < net.size(); ++i) {
		vector<Neuron> &prev = net[i-1];
		#ifdef DEBUG
		cout << "i = " << i << endl;
		#endif	
		for (int j = 0; j < net[i].size() -1; ++j) { // doesn't cal bias
			net[i][j].cal(prev);
		}
		net[i][net[i].size()-1].setOutput(1); //set bias
	}
}

void NN::backProp(const vector<double>& samp_output) {
	vector<Neuron> &last = net[net.size()-1];
	if (samp_output.size() != last.size() - 1) throw "NN size error";

	// calculate error
	double error = 0;
	for (int i = 0; i < last.size() - 1; ++i) {
		double a = samp_output[i] - last[i].getOutput();
		error += a*a;
	}
	error /= 2; //get loss or Ein

	// calculate output layer gradient
	for (int i = 0; i < last.size() - 1; ++i) {
		last[i].calOutputGradient(samp_output[i]);
		#ifdef DEBUG
		cout << "gradient is " << last[i].getGradient() << endl;
		#endif	
	}

	//calculate hidden later gradient
	for (int i = net.size() - 2; i > 0; --i) { // last two ~ two
		vector<Neuron> &next = net[i+1];
		vector<Neuron> &now = net[i];
		for (int j = 0; j < now.size() - 1; ++j) {
			now[j].calHiddenGradient(next, j);
			#ifdef DEBUG
			cout << "gradient is " << now[j].getGradient() << endl;
			#endif	
		}
	}

	// update all weights
	for (int i = net.size() - 1; i > 0; --i) {
		vector<Neuron> &prev = net[i-1];
		vector<Neuron> &now = net[i];
		for (int j = 0; j < now.size() - 1; ++j) {
			now[j].update(prev);
			#ifdef DEBUG
			now[j].printWeight();
			#endif	
		}
	}
}

void NN::getResult(const vector<double>& input, vector<double>& ans) {
	forward(input);
	int j = net.size() - 1;
	ans.clear();
	for (int i = 0; i < net[j].size() - 1; ++i) {
		ans.push_back(net[j][i].getOutput());
	}
}

double NN::calStandardError() {
	double error = 0;
	double count = 0;
	for (int i = 1; i < net.size(); ++i) {
		for (int j = 0; j < net[i].size() - 1; ++j) {
			count += net[i-1].size();
			error += net[i][j].calSquareError();
		}
	}
	return sqrt(error / count);
}

ostream& operator<<(ostream& os, const NN& nn) {
	// { int layer_num, { int neuron_num, Neuron[neuron_num] }[layer_num] }
	BinaryStream bs;
	// write obj to stream
	bs.writeInt(os, nn.net.size());
	for (int i = 0; i < nn.net.size(); i++) {
		bs.writeInt(os, nn.net[i].size());
		for (int j = 0; j < nn.net[i].size(); j++)
			os << nn.net[i][j];
	}
	return os;
}

istream& operator>>(istream& is, NN& nn) {
	// { int layer_num, { int neuron_num, Neuron[neuron_num] }[layer_num] }
	BinaryStream bs;
	// read obj from stream
	int layer_num = bs.readInt(is);
	for (int i = 0; i < layer_num; i++) {
		vector<Neuron> layer;
		int neuron_num = bs.readInt(is);
		for (int j = 0; j < neuron_num; j++) {
			Neuron n;
			is >> n;
			layer.push_back(n);
		}
		nn.net.push_back(layer);
	}
	return is;
}

void NN::print() {
	for (int i = 0; i < net.size(); ++i)
		for (int j = 0; j < net[i].size(); ++j)
			net[i][j].print();
		cout << endl;
}