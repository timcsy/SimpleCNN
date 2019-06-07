#include "../src/Kernel.hpp"
#include "../src/Convolution.hpp"
#include "../src/BinaryStream.hpp"
#include "../src/NN.hpp"
#include "../src/Records.hpp"
#include "../src/util.hpp"
#include <fstream>
#include <iomanip>
using namespace std;

void test_kernel() {
	Kernel k(3, 3);
	k.print();
	
	fstream fout("test/data/tmp/c_k_output.txt", ios::out);
	fout << k;
	fout.close();
	fstream fin("test/data/tmp/c_k_output.txt", ios::in);
	Kernel kk;
	fin >> kk;
	fin.close();
	kk.print();
}

void test_conv() {
	Layers map = {
		{
			{1,2,3,4,5,6,7},
			{8,9,10,11,12,13,14},
			{15,16,17,18,19,10,21},
			{22,23,24,25,26,27,28},
			{29,30,31,32,33,34,35},
			{36,37,38,39,40,41,42},
			{43,44,45,46,47,48,49}
		},
		{
			{-1,2,3,4,5,6,7},
			{8,9,10,11,12,13,14},
			{15,16,17,18,19,10,21},
			{22,23,24,25,26,27,28},
			{29,30,31,32,33,34,35},
			{36,37,38,39,40,41,42},
			{43,44,45,46,47,48,0}
		}
	};
	Convolution convolution(map, 3, 3, 3, 1, 1);

	cout << "kernels:" << endl;
	convolution.print_kernels();

	convolution.conv(); // with relu
	// convolution.conv(false); // not with relu
	cout << "convolution map:" << endl;
	convolution.print_conv();

	convolution.max_pooling(2, 2);
	cout << "pooling map:" << endl;
	convolution.print_pooling();

	convolution.flatten();
	cout << "flatten vector:" << endl;
	convolution.print_flatten();
}

void test_bs_1() {
	BinaryStream bs;
	bs.setDouble(cin);
	cout << setprecision(100) << bs.getDouble() << endl;
	fstream fout("test/data/tmp/bs_test1_output.txt", ios::out);
	bs.write(fout);
	fout.close();
	fstream fin("test/data/tmp/bs_test1_output.txt", ios::in);
	double rr = bs.readDouble(fin);
	cout << setprecision(100) << rr << endl;
}

void test_bs_2() {
	BinaryStream bs;
	int n = 8;
	double d = n;
	bs.setDouble(d);
	cout << setprecision(100) << bs.getDouble() << endl;
	fstream fout("test/data/tmp/bs_test2_output.txt", ios::out);
	bs.write(fout);
	fout.close();
	fstream fin("test/data/tmp/bs_test2_output.txt", ios::in);
	double rr = bs.readDouble(fin);
	cout << setprecision(100) << rr << endl;
}

void test_bs_endian() {
	cout << "Big-Endian = " << BinaryStream::isBigEndian() << endl;
}

void test_neuron_1() {
	Neuron n_out(10, 0.1);
	n_out.print();
	
	fstream fout("test/data/tmp/test_neuron_1_output.txt", ios::out);
	fout << n_out;
	fout.close();
	fstream fin("test/data/tmp/test_neuron_1_output.txt", ios::in);
	Neuron n_in;
	fin >> n_in;
	fin.close();

	n_in.print();
}

void test_neuron_2() {
	Neuron n_out({2, 3, 4, 5, 6}, 0.8);
	n_out.print();
	
	fstream fout("test/data/tmp/test_neuron_2_output.txt", ios::out);
	fout << n_out;
	fout.close();
	fstream fin("test/data/tmp/test_neuron_2_output.txt", ios::in);
	Neuron n_in;
	fin >> n_in;
	fin.close();

	n_in.print();
}

void test_nn_1() {
	// simple test
	vector<double> b({0.15, 0.2, 0.35}), b1({0.25, 0.3, 0.35}), c({0.4, 0.45, 0.6}), c1({0.5, 0.55, 0.6});
	Layers weights;
	vector<vector<double> > w, w2;
	w.push_back(b); w.push_back(b1);
	w2.push_back(c); w2.push_back(c1);
	weights.push_back(w); weights.push_back(w2);
	NN nn(weights, 5e-4);
	vector<double> input({0.05, 0.1}), output({0.01, 0.99});
	nn.forward(input);
	nn.backProp(output);
	cout << nn.calStandardError() << endl;
}

void test_nn_2() {
	// test weights
	Layers weights = {
		{ {0.15, 0.2, 0.35}, {0.25, 0.3, 0.35} },
		{ {0.4, 0.45, 0.6}, {0.5, 0.55, 0.6} }
	};
	NN nn(weights, 5e-4);
	vector<double> input({0.05, 0.1}), output({0.01, 0.99});
	nn.forward(input);
	nn.backProp(output);
	cout << nn.calStandardError() << endl;
}

void test_nn_3() {
	vector<int> shape {20, 3, 2};
	NN nn(shape, 5e-3, 0.5);

	Records train("test/data/tttrain.txt", " ", 20);
	train.read_label("test/data/ttlabel.txt");
	Records test("test/data/ttest.txt", " ", 20);
	test.setLabelMap(train);

	int count = 0;
	double min = 100;
	while (1) {
		nn.forward(train[count].data);
		nn.backProp(train[count].output);
		count++;
		if (count % train.size() == 0) {
			double now = nn.calStandardError();
			if(now < min) {
				min = now;
				// cout << "min = " << now << endl;
			}
			if(now <= nn.eps) break;
			count %= train.size();
			cout << now << endl;
		}
	}
	
	double err_num = 0;
	for (int i = 0; i < test.size(); ++i) {
		vector<double> a = nn.getResult(test[i].data);
		int ans = argmax(a);
		if (ans != test[i].id) err_num++;
	}

	nn.print();
	cout << "Eout = " << err_num / test.size() << endl;
}

void test_nn_4() {
	vector<vector<double> > shape {{20}, {3, 0.6}, {2, 0.4}};
	NN nn(shape, 5e-3);

	Records train("test/data/tttrain.txt", " ", 20);
	train.read_label("test/data/ttlabel.txt");
	Records test("test/data/ttest.txt", " ", 20);
	test.setLabelMap(train);

	int count = 0;
	double min = 100;
	while (1) {
		nn.forward(train[count].data);
		nn.backProp(train[count].output);
		count++;
		if (count % train.size() == 0) {
			double now = nn.calStandardError();
			if(now < min) {
				min = now;
				// cout << "min = " << now << endl;
			}
			if(now <= nn.eps) break;
			count %= train.size();
			cout << now << endl;
		}
	}
	
	double err_num = 0;
	for (int i = 0; i < test.size(); ++i) {
		vector<double> a = nn.getResult(test[i].data);
		int ans = argmax(a);
		if (ans != test[i].id) err_num++;
	}

	nn.print();
	cout << "Eout = " << err_num / test.size() << endl;
}

int main() {
	try {

		setup(); // must appear just once in main function
		// test_kernel();
		// test_conv();
		// test_bs_1();
		// test_bs_2();
		// test_bs_endian();
		// test_neuron_1();
		// test_neuron_2();
		// test_nn_1();
		// test_nn_2();
		test_nn_3();
		// test_nn_4();

	} catch (char const* s) {
		cout << s<< endl;
	}
	return 0;
}