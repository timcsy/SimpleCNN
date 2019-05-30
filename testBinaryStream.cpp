#include "BinaryStream.hpp"
#include <iomanip>
#include <fstream>
using namespace std;

void test1() {
	BinaryStream bs;
	bs.setDouble(cin);
	cout << setprecision(100) << bs.getDouble() << endl;
	fstream fout("bs_test1_output.txt", ios::out);
	bs.write(fout);
	fout.close();
	fstream fin("bs_test1_output.txt", ios::in);
	double rr = bs.readDouble(fin);
	cout << setprecision(100) << rr << endl;
}

void test2() {
	BinaryStream bs;
	int n = 8;
	double d = n;
	bs.setDouble(d);
	cout << setprecision(100) << bs.getDouble() << endl;
	fstream fout("bs_test2_output.txt", ios::out);
	bs.write(fout);
	fout.close();
	fstream fin("bs_test2_output.txt", ios::in);
	double rr = bs.readDouble(fin);
	cout << setprecision(100) << rr << endl;
}

void test_endian() {
	cout << "Big-Endian = " << BinaryStream::isBigEndian() << endl;
}

int main() {
	// test1();
	// test2();
	// test_endian();
	return 0;
}