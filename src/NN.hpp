#ifndef NN_H
#define NN_H

#include "Neuron.hpp"
using namespace std;

#define Epsilon 1e-3

class NN {
public:
	NN(const vector<int>& arr);
	NN(const vector<int>& arr, const vector<vector<vector<double> > >& w);
	void forward(const vector<double>& input);
	void backProp(const vector<double>& samp_output);
	void getResult(const vector<double>& input, vector<double>& ans);
	double calStandardError(){
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
	void print() {
		for (int i = 0; i < net.size(); ++i)
			for (int j = 0; j < net[i].size(); ++j)
				net[i][j].printWeight();
	}
private:
	vector<vector<Neuron> > net;

};

NN::NN(const vector<int>& arr) {
	for (int i = 0; i < arr.size(); ++i) {
		vector<Neuron> a;
		for (int j = 0; j <= arr[i]; ++j) { // include bias
			if (i == 0) {
				Neuron b(0);
				a.push_back(b);
			} else {
				Neuron b(arr[i-1]);
				a.push_back(b);
			}
		}
		net.push_back(a);
	}
}

NN::NN(const vector<int>& arr, const vector<vector<vector<double> > >& w) {
	for (int i = 0; i < arr.size(); ++i) {
		vector<Neuron> a;
		for (int j = 0; j < arr[i]; ++j) { // include bias
			// cout<<"i = "<<i<<endl;
			if (i == 0) {
				Neuron b(0);
				a.push_back(b);
			} else {
				Neuron b(w[i-1][j]);
				a.push_back(b);
			}
		}
		if (i == 0) {
			Neuron b(0);
			a.push_back(b);
		} else {
			Neuron b(arr[i-1]);
			a.push_back(b);
		}
		net.push_back(a);
	}
}

void NN::forward(const vector<double>& input) {
	assert(input.size() == net[0].size() - 1);
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
	assert(samp_output.size() == last.size() - 1);

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

void read(char * filename, vector<vector<double> >& data, vector<vector<double> >& y) {
	FILE *f;
	if ((f = fopen(filename, "r")) != NULL){
		while (!feof(f)) {
			double t;
			vector<double> a;
			vector<double> b;
			for (int i = 0; i < 20; ++i){
				fscanf(f, "%lf", &t); // X1~X20
				a.push_back(t);
			}
			fscanf(f, "%lf", &t);
			if(t == 1)
				b.push_back(t);
			else if(t == -1)
				b.push_back(0);
			data.push_back(a);
			y.push_back(b);
		}
	}
}

void print(vector<double> a) {
	for (int i = 0; i < a.size(); ++i)
		cout << a[i] << " ";
	cout << endl;
}

#endif