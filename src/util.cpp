#include "util.hpp"
using namespace std;

double sigmoid(double x) {
	return (1 / (1 + exp(-x)));
}

void read(char * filename, vector<vector<double> >& data, vector<vector<double> >& y) {
	FILE *f;
	if ((f = fopen(filename, "r")) != NULL){
		while (!feof(f)) {
			double t;
			vector<double> a;
			vector<double> b;
			for (int i = 0; i < 20; ++i){
				fscanf(f, "%lf", &t); // X1~X20
				a.push_back(t);
			}
			fscanf(f, "%lf", &t);
			if(t == 1)
				b.push_back(t);
			else if(t == -1)
				b.push_back(0);
			data.push_back(a);
			y.push_back(b);
		}
	}
}