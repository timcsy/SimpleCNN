#include <iostream>
using namespace std;

class BinaryStream {
public:
	double getInt() { return bs.n; }
	void getInt(ostream& os) { os << bs.n; }
	void setInt(int n) { clear(); bs.n = n; }
	void setInt(istream& is) { clear(); is >> bs.n; }
	double getDouble() { return bs.r; }
	void getDouble(ostream& os) { os << bs.r; }
	void setDouble(double r) { clear(); bs.r = r; }
	void setDouble(istream& is) { clear(); is >> bs.r; }
	void read(istream& is) {
		if (isBigEndian()) {
			is.read(getBuf(), 8);
		} else {
			for (int i = 7; i >=0; i--) {
				is.read(getBuf() + i, 1);
			}
		}
	}
	void write(ostream& os) {
		if (isBigEndian()) {
			os.write(getBuf(), 8);
		} else {
			for (int i = 7; i >=0; i--) {
				os.write(getBuf() + i, 1);
			}
		}
	}
	int readInt(istream& is) {
		read(is);
		return getInt();
	}
	void writeInt(ostream& os, int n) {
		setInt(n);
		write(os);
	}
	double readDouble(istream& is) {
		read(is);
		return getDouble();
	}
	void writeDouble(ostream& os, double r) {
		setDouble(r);
		write(os);
	}
	static bool isBigEndian() {
		BinaryStream bs;
		bs.setInt(1);
		if (bs.getBuf()[0] > 0) return false;
		else return true;
	}
private:
	union {
		int n;
		double r;
		char s[8];
	} bs;
	char * getBuf() { return bs.s; }
	void clear() {
		for (int i = 0; i < 8; i++) bs.s[i] = 0;
	}
};