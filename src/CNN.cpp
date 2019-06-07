#include "CNN.hpp"
using namespace std;

CNN::CNN(Layers config) {
	// map
	int map_height = (config[0][0].size() > 0)? config[0][0][0]: 0;
	int map_width = (config[0][0].size() > 1)? config[0][0][1]: 0;
	int map_depth = (config[0][0].size() > 2)? config[0][0][2]: 1;
	// convolution Layers
	for (int l = 1; l < config[0].size(); l++) {
		int kernel_num = (config[0][l].size() > 0)? config[0][l][0]: 0;
		int kernek_height = (config[0][l].size() > 1)? config[0][l][1]: 4;
		int kernel_weight = (config[0][l].size() > 2)? config[0][l][2]: 4;
		int stride = (config[0][l].size() > 3)? config[0][l][3]: 1;
		int padding = (config[0][l].size() > 4)? config[0][l][4]: 0;
		bool relu = (config[0][l].size() > 5)? config[0][l][5]: true;
		int pooling_height = (config[0][l].size() > 6)? config[0][l][6]: 2;
		int pooling_width = (config[0][l].size() > 7)? config[0][l][7]: 2;
		Convolution layer(kernel_num, kernek_height, kernel_weight, stride, padding, relu, pooling_height, pooling_width);
		conv_layers.push_back(layer);
	}
	// calculate convolution output number
	Layers input;
	for (int l = 0; l < map_depth; l++) {
		vector<vector<double> > layer;
		for (int i = 0; i < map_height; i++) {
			vector<double> row;
			for (int j = 0; j < map_width; j++) {
				row.push_back(0);
			}
			layer.push_back(row);
		}
		input.push_back(layer);
	}
	for (int l = 0; l < conv_layers.size(); l++) {
		conv_layers[l].feed(input);
		conv_layers[l].conv();
		input = conv_layers[l].max_pooling();
	}
	vector<double> output = conv_layers[conv_layers.size()-1].flatten();
	int NN_input_size = output.size();
	// NN
	double eps = config[1][0][0];
	int N = config[1][0][1];
	config[1][0] = vector<double>(1, NN_input_size);
	nn = NN(config[1], eps, N);
}