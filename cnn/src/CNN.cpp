#include "CNN.hpp"
using namespace std;

CNN::CNN(Layers config, vector<string> labels): labels(labels) {
	setConvolution(config[0]);
	setNN(config[1]);
}

void CNN::setConvolution(Config config) {
	// map
	map_height = (config[0].size() > 0)? config[0][0]: 0;
	map_width = (config[0].size() > 1)? config[0][1]: 0;
	map_depth = (config[0].size() > 2)? config[0][2]: 1;
	// convolution Layers
	conv_layers.clear();
	for (int l = 1; l < config.size(); l++) {
		int kernel_num = (config[l].size() > 0)? config[l][0]: 0;
		int kernek_height = (config[l].size() > 1)? config[l][1]: 4;
		int kernel_weight = (config[l].size() > 2)? config[l][2]: 4;
		int stride = (config[l].size() > 3)? config[l][3]: 1;
		int padding = (config[l].size() > 4)? config[l][4]: 0;
		bool relu = (config[l].size() > 5)? config[l][5]: true;
		int pooling_height = (config[l].size() > 6)? config[l][6]: 2;
		int pooling_width = (config[l].size() > 7)? config[l][7]: 2;
		Convolution layer(kernel_num, kernek_height, kernel_weight, stride, padding, relu, pooling_height, pooling_width);
		conv_layers.push_back(layer);
	}
}

void CNN::setNN(Config config) {
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
	int NN_input_size = feed_conv(input).size();
	// NN
	double eps = config[0][0];
	int N = config[0][1];
	config[0] = vector<double>(1, NN_input_size);
	nn = NN(config, eps, N);
}

vector<double> CNN::feed_conv(Layers& input) {
	for (int l = 0; l < conv_layers.size(); l++) {
		conv_layers[l].feed(input);
		conv_layers[l].conv();
		input = conv_layers[l].max_pooling();
	}
	return conv_layers[conv_layers.size()-1].flatten();
}

Layers reshape(vector<double> v, int height, int width, int depth) {
	Layers maps;
	for (int l = 0; l < depth; l++) {
		vector<vector<double> > layer;
		for (int i = 0; i < height; i++) {
			vector<double> row;
			for (int j = 0; j < width; j++) {
				row.push_back(v[(l * depth + i) * height + j]);
			}
			layer.push_back(row);
		}
		maps.push_back(layer);
	}
	return maps;
}

string CNN::getResult(vector<double> input) {
	Layers data = reshape(input, map_height, map_width, map_depth);
	vector<double> res = nn.getResult(feed_conv(data));
	int ans = argmax(res);
	return labels[ans];
}

Records CNN::conv(Records& train_data, bool show) {
	Records flatten_data;
	flatten_data.setLabelMap(train_data);
	for (int i = 0; i < train_data.size(); i++) {
		Record rec;
		rec.label = train_data[i].label;
		rec.id = train_data[i].id;
		rec.output = train_data[i].output;
		Layers data = reshape(train_data[i].data, map_height, map_width, map_depth);
		rec.data = feed_conv(data);
		flatten_data.push_back(rec);
		if (show) cout << "convolution record num = " << i << endl;
	}
	return flatten_data;
}

double CNN::train_nn(Records& flatten_data, bool show) {
	return nn.train(flatten_data, show);
}

double CNN::train(Records& train_data, bool show) {
	Records flatten_data = conv(train_data, show);
	return train_nn(flatten_data, show);
}

double CNN::test_nn(Records& flatten_data, bool show) {
	return test(flatten_data, show);
}

double CNN::test(Records& test_data, bool show) {
	Records flatten_data = conv(test_data, show);
	return nn.test(flatten_data);
}

ostream& operator<<(ostream& os, const CNN& cnn) {
	// { int map_height, int map_width, int map_depth, int layer_size, Convolution[layer_size], nn }
	BinaryStream bs;
	// write obj to stream
	bs.writeInt(os, cnn.map_height);
	bs.writeInt(os, cnn.map_width);
	bs.writeInt(os, cnn.map_depth);
	bs.writeInt(os, cnn.conv_layers.size());
	for (int l = 0; l < cnn.conv_layers.size(); l++) {
		os << cnn.conv_layers[l];
	}
	os << cnn.nn;
	bs.writeInt(os, cnn.labels.size());
	for (int i = 0; i < cnn.labels.size(); i++) {
		os << cnn.labels[i] << endl;
	}
	return os;
}

istream& operator>>(istream& is, CNN& cnn) {
	// { int map_height, int map_width, int map_depth, int layer_size, Convolution[layer_size], nn, int label_num, string labels }
	BinaryStream bs;
	// read obj from stream
	cnn.map_height = bs.readInt(is);
	cnn.map_width = bs.readInt(is);
	cnn.map_depth = bs.readInt(is);
	int layer_size = bs.readInt(is);
	for (int l = 0; l < layer_size; l++) {
		Convolution c;
		is >> c;
		cnn.conv_layers.push_back(c);
	}
	is >> cnn.nn;
	int label_num = bs.readInt(is);
	cnn.labels.clear();
	for (int i = 0; i < label_num; i++) {
		string s;
		is >> s;
		cnn.labels.push_back(s);
	}
	return is;
}

void CNN::print() {
	for (int l = 0; l < conv_layers.size(); l++) {
		cout << "Convolution Layer " << l << ":" << endl;
		conv_layers[l].print();
	}
	nn.print();
}