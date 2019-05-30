#include <vector>
#include <cstdlib>
#include <ctime>
#include <iostream>
using namespace std;

typedef vector<vector<vector<double> > > Layers;

class Kernel {
public:
	Kernel(int height, int width);
	static void setup() { srand((unsigned)time(NULL)); } // put it once in main function (MUST DO !!!)
	vector<double>& operator[](int row) { return map[row]; }
	int getHeight() { return map.size(); }
	int getWidth() { if(map.size() > 0) return map[0].size(); return 0; }
	void print();
private:
	vector<vector<double> > map;
};

Kernel::Kernel(int height, int width) {
	map.resize(height);
	for (int i = 0; i < height; i++) {
		map[i].resize(width);
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			map[i][j] = rand() % 2;
		}
	}
}

void Kernel::print() {
	for (int i = 0; i < getHeight(); i++) {
		for (int j = 0; j < getWidth(); j++) {
			cout << map[i][j] << " ";
		}
		cout << endl;
	}
}

class Convolution {
public:
	Convolution(Layers map, Layers kernels, int strides = 1, int padding = 0):
		input_map(map), strides(strides), padding(padding) {}
	Convolution(Layers map, int kn, int kh = 4, int kw = 4, int strides = 1, int padding = 0);
	Layers conv(bool relu = true);
	Layers max_pooling(int m, int n);
	vector<double> flatten();
	void print_kernels();
	void print_conv();
	void print_pooling();
	void print_flatten();
private:
	Layers input_map;
	vector<Kernel> kernels;
	Layers conv_map;
	Layers pooling_map;
	vector<double> output_vector;
	int strides;
	int padding;
};

Convolution::Convolution(Layers map, int kn, int kh, int kw, int strides, int padding) {
	this->input_map = map;
	this->strides = strides;
	this->padding = padding;
	for (int i = 0; i < kn; i++) {
		kernels.push_back(Kernel(kh, kw)); // not checking if they are the same
	}
}

Layers Convolution::conv(bool relu) {
	// alias
	int H = 0, W = 0, KH = 0, KW = 0;
	if (input_map.size() > 0) {
		H = input_map[0].size();
		if (H > 0) W = input_map[0][0].size();
	}
	if (kernels.size() > 0) {
		KH = kernels[0].getHeight();
		KW = kernels[0].getWidth();
	}
	int height = (H + 2 * padding - KH) / strides + 1;
	int width = (W + 2 * padding - KW) / strides + 1;
	// initializing convolution map
	conv_map.resize(kernels.size());
	for (int k = 0; k < conv_map.size(); k++) {
		conv_map[k].resize(height);
		for (int i = 0; i < height; i++) {
			conv_map[k][i].resize(width, 0);
		}
	}
	// convolution
	for (int k = 0; k < kernels.size(); k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				for (int l = 0; l < input_map.size(); l++) {
					for (int p = 0; p < KH; p++) {
						for (int q = 0; q < KW; q++) {
							if (0 <= strides*i+p-padding && strides*i+p-padding < H &&
									0 <= strides*j+q-padding && strides*j+q-padding < W) {
								conv_map[k][i][j] += input_map[l][strides*i+p-padding][strides*j+q-padding] * kernels[k][p][q];
							}
						}
					}
				}
				if (relu) conv_map[k][i][j] = (conv_map[k][i][j] > 0)? conv_map[k][i][j]: 0;
			}
		}
	}
	return conv_map;
}

Layers Convolution::max_pooling(int m, int n) {
	// alias
	int CH = 0, CW = 0;
	if (conv_map.size() > 0) {
		CH = conv_map[0].size();
		if (CH > 0) CW = conv_map[0][0].size();
	}
	int height = (CH % m == 0)? (CH / m): (CH / m + 1);
	int width = (CW % n == 0)? (CW / n): (CW / n + 1);
	// initialization
	double min = 0;
	for (int k = 0; k < conv_map.size(); k++)
		for (int i = 0; i < conv_map[k].size(); i++)
			for (int j = 0; j < conv_map[k][i].size(); j++)
				if (conv_map[k][i][j] < min) min = conv_map[k][i][j];
	pooling_map.resize(conv_map.size());
	for (int k = 0; k < pooling_map.size(); k++) {
		pooling_map[k].resize(height);
		for (int i = 0; i < height; i++) {
			pooling_map[k][i].resize(width, min - 1);
		}
	}
	// max pooling
	for (int k = 0; k < pooling_map.size(); k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				for (int p = 0; p < m; p++) {
					for (int q = 0; q < n; q++) {
						if (0 <= m*i+p && m*i+p < CH && 0 <= n*j+q && n*j+q < CW) {
							if (pooling_map[k][i][j] < conv_map[k][m*i+p][n*j+q])
								pooling_map[k][i][j] = conv_map[k][m*i+p][n*j+q];
						}
					}
				}
			}
		}
	}
	return pooling_map;
}

vector<double> Convolution::flatten() {
	output_vector.clear();
	for (int k = 0; k < pooling_map.size(); k++) {
		for (int i = 0; i < pooling_map[k].size(); i++) {
			for (int j = 0; j < pooling_map[k][i].size(); j++) {
				output_vector.push_back(pooling_map[k][i][j]);
			}
		}
	}
	return output_vector;
}

void Convolution::print_kernels() {
	for (int k = 0; k < kernels.size(); k++) {
		kernels[k].print();
		cout << endl;
	}
}

void Convolution::print_conv() {
	for (int k = 0; k < conv_map.size(); k++) {
		for (int i = 0; i < conv_map[k].size(); i++) {
			for (int j = 0; j < conv_map[k][i].size(); j++) {
				cout << conv_map[k][i][j] << " ";
			}
			cout << endl;
		}
		cout << endl;
	}
}

void Convolution::print_pooling() {
	for (int k = 0; k < pooling_map.size(); k++) {
		for (int i = 0; i < pooling_map[k].size(); i++) {
			for (int j = 0; j < pooling_map[k][i].size(); j++) {
				cout << pooling_map[k][i][j] << " ";
			}
			cout << endl;
		}
		cout << endl;
	}
}

void Convolution::print_flatten() {
	for (int i = 0; i < output_vector.size(); i++)
		cout << output_vector[i] << " ";
	cout << endl;
}