#include "CNN.hpp"

int main(int argc, char * argv[]) {
	// read cnn
	CNN cnn;
	fstream fin(argv[1], ios::in);
	fin >> cnn;
	fin.close();

	// predict
	vector<double> input;
	double n;
	while (cin >> n) input.push_back(n);
	cout << cnn.getResult(input);
	return 0;
}