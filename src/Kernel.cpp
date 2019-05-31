#include "Kernel.hpp"

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

ostream& operator<<(ostream& os, const Kernel& k) {
	BinaryStream bs;
	// write obj to stream
	// alias
	int height = k.map.size(), width = 0;
	if (height > 0) width = k.map[0].size();
	bs.writeInt(os, height);
	bs.writeInt(os, width);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			bs.writeDouble(os, k.map[i][j]);
		}
	return os;
}

istream& operator>>(istream& is, Kernel& k) {
	BinaryStream bs;
	// read obj from stream
	int height = bs.readInt(is);
	int width = bs.readInt(is);
	for (int i = 0; i < height; i++) {
		vector<double> row;
		for (int j = 0; j < width; j++)
			row.push_back(bs.readDouble(is));
		k.map.push_back(row);
	}
	return is;
}

void Kernel::print() {
	for (int i = 0; i < getHeight(); i++) {
		for (int j = 0; j < getWidth(); j++) {
			cout << map[i][j] << " ";
		}
		cout << endl;
	}
}