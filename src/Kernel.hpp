#ifndef KERNEL_H
#define KERNEL_H

#include "BinaryStream.hpp"
#include <vector>
#include <cstdlib>
#include <ctime>
#include <iostream>
using namespace std;

class Kernel {
public:
	Kernel() {}
	Kernel(int height, int width);
	static void setup() { srand((unsigned)time(NULL)); } // put it once in main function (MUST DO !!!)
	vector<double>& operator[](int row) { return map[row]; }
	int getHeight() { return map.size(); }
	int getWidth() { if (map.size() > 0) return map[0].size(); return 0; }
	friend ostream& operator<<(ostream& os, const Kernel& k);
	friend istream& operator>>(istream& is, Kernel& k);
	void print();
private:
	vector<vector<double> > map;
};

#endif