#include <cstdio>
#include <vector>
#include <iostream>
#include <cassert>
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
			delta_weight.push_back(0);
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
	double getWeight(int i) const { assert(i<size); return weight[i]; }
	void cal(const vector<Neuron>& prev);
	void calOutputGradient(const double samp_output);
	void calHiddenGradient(const vector<Neuron>& next, int j);
	void update(const vector<Neuron>& prev);
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

class Net {
public:
	Net(const vector<int>& arr);
	Net(const vector<int>& arr, const vector<vector<vector<double> > >& w);
	void forward(const vector<double>& input);
	void backProp(const vector<double>& samp_output);
	void getResult(const vector<double>& input, vector<double>& ans);
	void print() {
		for (int i = 0; i < net.size(); ++i)
			for (int j = 0; j < net[i].size(); ++j)
				net[i][j].printWeight();
	}
private:
	vector<vector<Neuron> > net;
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


Net::Net(const vector<int>& arr) {
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
Net::Net(const vector<int>& arr, const vector<vector<vector<double> > >& w) {
	for (int i = 0; i < arr.size(); ++i) {
		vector<Neuron> a;
		for (int j = 0; j < arr[i]; ++j) { // include bias
			// cout<<"i = "<<i<<endl;
			if (i == 0) {
				Neuron b(0);
				a.push_back(b);
			} else {
				Neuron b(arr[i-1], w[i-1][j]);
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
void Net::forward(const vector<double>& input) {
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
void Net::backProp(const vector<double>& samp_output) {
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
void Net::getResult(const vector<double>& input, vector<double>& ans) {
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
	for (int i = 0; i <a.size(); ++i)
		cout << a[i] << " ";
	cout << endl;
}

int main() {
	vector<int> a {20, 10, 1};
	// vector<double> b{1,1},c1{1,1,1.2},c2{1,1,-0.3},c3{0.4,0.8,-0.5};
	// vector<double> y{0};
	vector<vector<vector<double> > > weight;
	// vector<vector<double>> w,w2;w.push_back(c1);w.push_back(c2);
	// w2.push_back(c3);
	// weight.push_back(w);weight.push_back(w2);

	vector<vector<double> > trainData,testData; 
	vector<vector<double> > trainY,testY;
	char s[] = "tttrain.txt";
	read(s,trainData,trainY);
	char c[] = "ttest.txt";
	read(c,testData,testY);

	Net mynet(a);
	for (int i = 0; i < 20000; ++i){
		mynet.forward(trainData[i%trainData.size()]);
		mynet.backProp(trainY[i%trainY.size()]);
	}
	double ans = 0;
	for (int i = 0; i < testData.size(); ++i) {
		vector<double> a;
		mynet.getResult(testData[i], a);
		double aaa;
		if (a.at(0) >= 0.5) {
			aaa = 1;
		} else
			aaa = 0;

		if(aaa != testY[i][0])
			ans++;
	}
	mynet.print();
	cout << ans / testData.size();
	// vector<double> Ans;
	// mynet.getResult(b,Ans);
	// for(int i = 0;i<Ans.size();++i)
	// 	cout<<Ans[i]<<" ";
}