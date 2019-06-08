#include "../src/Kernel.hpp"
#include "../src/Convolution.hpp"
#include "../src/BinaryStream.hpp"
#include "../src/NN.hpp"
#include "../src/Records.hpp"
#include "../src/CNN.hpp"
#include <fstream>
#include <iomanip>
using namespace std;

void test_kernel() {
	Kernel k(3, 3);
	k.print();
	
	fstream fout("tmp/c_k_output.txt", ios::out);
	fout << k;
	fout.close();
	fstream fin("tmp/c_k_output.txt", ios::in);
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
	Convolution convolution(3, 3, 3, 1, 1, true, 2, 2);

	cout << "input map:" << endl;
	convolution.feed(map);

	cout << "kernels:" << endl;
	convolution.print_kernels();

	convolution.conv(); // with relu
	// convolution.conv(false); // not with relu
	cout << "convolution map:" << endl;
	convolution.print_conv();

	convolution.max_pooling();
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
	fstream fout("tmp/bs_test1_output.txt", ios::out);
	bs.write(fout);
	fout.close();
	fstream fin("tmp/bs_test1_output.txt", ios::in);
	double rr = bs.readDouble(fin);
	cout << setprecision(100) << rr << endl;
}

void test_bs_2() {
	BinaryStream bs;
	int n = 8;
	double d = n;
	bs.setDouble(d);
	cout << setprecision(100) << bs.getDouble() << endl;
	fstream fout("tmp/bs_test2_output.txt", ios::out);
	bs.write(fout);
	fout.close();
	fstream fin("tmp/bs_test2_output.txt", ios::in);
	double rr = bs.readDouble(fin);
	cout << setprecision(100) << rr << endl;
}

void test_bs_endian() {
	cout << "Big-Endian = " << BinaryStream::isBigEndian() << endl;
}

void test_records() {
	Records records("test/data/records.csv", ",", "b");
	for (int i = 0; i < records.size(); i++) {
		for (int j = 0; j < records[i].data.size(); j++) {
			cout << records[i].data[j] << " ";
		}
		cout << endl;
	}
}

void test_neuron_1() {
	Neuron n_out(10, 0.1);
	n_out.print();
	
	fstream fout("tmp/test_neuron_1_output.txt", ios::out);
	fout << n_out;
	fout.close();
	fstream fin("tmp/test_neuron_1_output.txt", ios::in);
	Neuron n_in;
	fin >> n_in;
	fin.close();

	n_in.print();
}

void test_neuron_2() {
	Neuron n_out({2, 3, 4, 5, 6}, 0.8);
	n_out.print();
	
	fstream fout("tmp/test_neuron_2_output.txt", ios::out);
	fout << n_out;
	fout.close();
	fstream fin("tmp/test_neuron_2_output.txt", ios::in);
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
	vector<int> shape {2, 2, 2};
	NN nn(shape, 5e-3, 0, 0.5);

	Records train_data("test/data/hw4/hw4_train.dat", " ", 2);
	train_data.setLabel({"1", "-1"});
	Records test_data("test/data/hw4/hw4_test.dat", " ", 2);
	test_data.setLabelMap(train_data);

	double Ein = nn.train(train_data, true);
	double Eout = nn.test(test_data);

	nn.print();
	cout << "Ein = " << Ein << endl;
	cout << "Eout = " << Eout << endl;
}

void test_nn_4() {
	vector<vector<double> > shape {{20}, {5, 0.6}, {2, 0.4}};
	NN nn(shape, 0, 1000);

	Records train_data("test/data/nn/tttrain.txt", " ", 20);
	train_data.setLabel({"1", "-1"});
	Records test_data("test/data/nn/ttest.txt", " ", 20);
	test_data.setLabelMap(train_data);

	double Ein = nn.train(train_data, true);
	double Eout = nn.test(test_data);

	nn.print();
	cout << "Ein = " << Ein << endl;
	cout << "Eout = " << Eout << endl;
}

void test_cnn_1() {
	vector<string> labels = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
	Records train_data("test/data/digit-recognizer/train.csv", ",", "label");
	train_data.setLabel(labels);
	Records test_data("test/data/digit-recognizer/test.csv", ",", "label");
	test_data.setLabelMap(train_data);

	Layers config = {
		{
			{28, 28, 1}, // map_height, map_width, map_depth
			{7, 4, 4, 1, 0, 1, 4, 4}, // kernel_num, kernek_height, kernel_weight, stride, padding, relu, pooling_height, pooling_width
			{7, 2, 2, 1, 0, 1, 4, 4}
		},
		{
			{0, 100}, // eps, N
			{20, 0.5}, // hidden_layer, learning_rate
			{10, 0.5} // output_layer, learning_rate
		}
	};

	CNN cnn(config, labels);

	double Ein = cnn.train(train_data, true);
	double Eout = cnn.test(test_data, true);

	fstream fout("tmp/test_cnn_1.cnn", ios::out);
	fout << cnn;
	fout.close();

	cnn.print();
	cout << "Ein = " << Ein << endl;
	cout << "Eout = " << Eout << endl;
}

void test_cnn_2() {
	// save CNN model
	Layers config = {
		{
			{28, 28, 1}, // map_height, map_width, map_depth
			{7, 4, 4, 1, 0, 1, 4, 4}, // kernel_num, kernek_height, kernel_weight, stride, padding, relu, pooling_height, pooling_width
			{7, 2, 2, 1, 0, 1, 4, 4}
		},
		{
			{0, 100}, // eps, N
			{20, 0.5}, // hidden_layer, learning_rate
			{10, 0.5} // output_layer, learning_rate
		}
	};
	vector<string> labels = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};

	CNN cnn(config, labels);

	fstream fout("tmp/test_cnn_2.cnn", ios::out);
	fout << cnn;
	fout.close();

	cnn.print();
}

void test_cnn_3() {
	// save convolutioned records
	fstream fin;
	fstream fout;

	CNN cnn;
	fin.open("tmp/test_cnn_2.cnn", ios::in);
	fin >> cnn;
	fin.close();

	Records train_data("test/data/digit-recognizer/train.csv", ",", "label");
	train_data.setLabel({"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"});

	fout.open("tmp/test_cnn_3_train.rec", ios::out);
	fout << cnn.conv(train_data, true);
	fout.close();

	Records test_data("test/data/digit-recognizer/test.csv", ",", "label");
	test_data.setLabelMap(train_data);

	fout.open("tmp/test_cnn_3_test.rec", ios::out);
	fout << cnn.conv(test_data, true);
	fout.close();
}


void test_cnn_4() {
	// use the saved convolutioned records
	fstream fin;

	CNN cnn;
	fin.open("tmp/test_cnn_2.cnn", ios::in);
	fin >> cnn;
	fin.close();

	// We can change the NN part only
	Config config = {
		{0, 100}, // eps, N
		{20, 0.5}, // hidden_layer, learning_rate
		{10, 0.5} // output_layer, learning_rate
	};
	cnn.setNN(config);

	Records flatten_train_data;
	fin.open("tmp/test_cnn_3_train.rec", ios::in);
	fin >> flatten_train_data;
	fin.close();

	Records flatten_test_data;
	fin.open("tmp/test_cnn_3_test.rec", ios::in);
	fin >> flatten_test_data;
	fin.close();

	double Ein = cnn.train_nn(flatten_train_data, true);
	double Eout = cnn.test_nn(flatten_test_data, true);

	fstream fout("tmp/test_cnn_1.cnn", ios::out);
	fout << cnn;
	fout.close();

	cnn.print();
	cout << "Ein = " << Ein << endl;
	cout << "Eout = " << Eout << endl;
}

int main() {
	try {

		setup(); // must appear just once in main function
		// test_kernel();
		// test_conv();
		// test_bs_1();
		// test_bs_2();
		// test_bs_endian();
		// test_records();
		// test_neuron_1();
		// test_neuron_2();
		// test_nn_1();
		// test_nn_2();
		// test_nn_3();
		// test_nn_4();
		// test_cnn_1();
		test_cnn_2();
		// test_cnn_3();
		// test_cnn_4();

	} catch (char const * s) {
		cout << s << endl;
	}
	return 0;
}