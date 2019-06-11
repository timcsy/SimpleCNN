#include "Convolution.hpp"

Convolution::Convolution(int kn, int kh, int kw, int strides, int padding, bool relu, int ph, int pw) {
	this->strides = strides;
	this->padding = padding;
	this->relu = relu;
	this->pooling_height = ph;
	this->pooling_width = pw;
	for (int i = 0; i < kn; i++) {
		kernels.push_back(Kernel(kh, kw)); // not checking if they are the same
	}
}

Layers Convolution::conv() {
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
	conv_map.clear();
	for (int k = 0; k < kernels.size(); k++) {
		vector<vector<double> > layer;
		for (int i = 0; i < height; i++) {
			vector<double> row;
			for (int j = 0; j < width; j++) {
				row.push_back(0);
			}
			layer.push_back(row);
		}
		conv_map.push_back(layer);
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
			}
		}
	}
	for (int k = 0; k < kernels.size(); k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (relu) conv_map[k][i][j] = (conv_map[k][i][j] > 0)? conv_map[k][i][j]: 0;
			}
		}
	}
	return conv_map;
}

Layers Convolution::max_pooling() {
	// alias
	int m = pooling_height, n = pooling_width;
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
	pooling_map.clear();
	for (int k = 0; k < conv_map.size(); k++) {
		vector<vector<double> > layer;
		for (int i = 0; i < height; i++) {
			vector<double> row;
			for (int j = 0; j < width; j++) {
				row.push_back(min - 1);
			}
			layer.push_back(row);
		}
		pooling_map.push_back(layer);
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

ostream& operator<<(ostream& os, const Convolution& c) {
	// { int strides, int padding, int relu, int pooling_height, int pooling_width, int kernel_num, Kernel[kernel_num] }
	BinaryStream bs;
	// write obj to stream
	bs.writeInt(os, c.strides);
	bs.writeInt(os, c.padding);
	bs.writeInt(os, c.relu);
	bs.writeInt(os, c.pooling_height);
	bs.writeInt(os, c.pooling_width);
	bs.writeInt(os, c.kernels.size());
	for (int i = 0; i < c.kernels.size(); i++)
			os << c.kernels[i];
	return os;
}

istream& operator>>(istream& is, Convolution& c) {
	// { int strides, int padding, int relu, int pooling_height, int pooling_width, int kernel_num, Kernel[kernel_num] }
	BinaryStream bs;
	// read obj from stream
	c.strides = bs.readInt(is);
	c.padding = bs.readInt(is);
	c.relu = bs.readInt(is);
	c.pooling_height = bs.readInt(is);
	c.pooling_width = bs.readInt(is);
	int kernel_num = bs.readInt(is);
	for (int i = 0; i < kernel_num; i++) {
		Kernel k;
		is >> k;
		c.kernels.push_back(k);
	}
	return is;
}

void Convolution::print() {
	cout << "strides = " << strides << endl;
	cout << "padding = " << padding << endl;
	cout << "relu = " << relu << endl;
	cout << "pooling_height = " << pooling_height << endl;
	print_kernels();
}

void Convolution::print_input() {
	for (int k = 0; k < input_map.size(); k++) {
		for (int i = 0; i < input_map[k].size(); i++) {
			for (int j = 0; j < input_map[k][i].size(); j++) {
				cout << input_map[k][i][j] << " ";
			}
			cout << endl;
		}
		cout << endl;
	}
}

void Convolution::print_kernels() {
	for (int k = 0; k < kernels.size(); k++) {
		cout << "Kernel " << k << ":" << endl;
		kernels[k].print();
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