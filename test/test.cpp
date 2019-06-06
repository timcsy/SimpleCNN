#include "../src/Kernel.hpp"
#include "../src/Convolution.hpp"
#include "../src/BinaryStream.hpp"
#include "../src/NN.hpp"
#include <fstream>
#include <iomanip>
using namespace std;

#define eps 5e-4

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
	vector<int> a({2, 2, 2});
	NN nn(a, weights);
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
	vector<int> a({2, 2, 2});
	NN nn(a, weights);
	vector<double> input({0.05, 0.1}), output({0.01, 0.99});
	nn.forward(input);
	nn.backProp(output);
	cout << nn.calStandardError() << endl;
}

void test_nn_3() {
	vector<int> a {20, 1, 1};
	vector<vector<double> > trainData, testData;
	vector<vector<double> > trainY, testY;
	char s[] = "test/data/tttrain.txt";
	read(s, trainData, trainY);
	char c[] = "test/data/ttest.txt";
	read(c, testData, testY);

	NN mynet(a);

	int count = 0;
	double min = 100;
	while(1){
		mynet.forward(trainData[count]);
		mynet.backProp(trainY[count]);
		count++;
		if(count % trainData.size() == 0){
			double now = mynet.calStandardError();
			if(now < min){
				min = now;
				// cout<<"min = "<<now<<endl;
			}
			if( now <= eps){
				break;
			}
			count %= trainData.size();
			cout<<now<<"\n";
		}
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

int main() {
	Kernel::setup(); // must appear just once in main function
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
	return 0;
}