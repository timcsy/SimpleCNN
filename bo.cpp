#include <iostream>
#include <iomanip>
#include <fstream>
using namespace std;

class CharBuf {
public:
	double getDouble() { return cb.r; }
	void setDouble(double r) { cb.r = r; }
	char * getBuf() { return cb.s; }
private:
	union {
		double r;
		char s[8];
	} cb;
};

int main() {
	CharBuf cb;
	double r;
	cin >> r;
	cb.setDouble(r);
	cout << setprecision(100) << cb.getDouble() << endl;
	fstream fout("output.txt", ios::out);
	fout.write(cb.getBuf(), sizeof(double));
	fout.close();
	fstream fin("output.txt", ios::in);
	fin.read(cb.getBuf(), 8);
	cout << setprecision(100) << cb.getDouble() << endl;
	return 0;
}