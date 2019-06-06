#ifndef UTIL_H
#define UTIL_H

#include <vector>
#include <cmath>
#include <string>
using namespace std;

typedef vector<vector<vector<double> > > Layers;

struct Record {
	vector<double> data;
	string label;
	Record() {}
	Record(vector<double> data, string label): data(data), label(label) {}
};

void setup(); // put it once in main function for random (MUST DO !!!)
double sigmoid(double x);
vector<string> split(const string s, const string delim);
vector<Record> read_csv(const string filename, const string delim, int col = 0, bool line_trim = true);
vector<Record> read_csv(const string filename, const string delim, const string label, bool line_trim = true);
vector<vector<double> > getData(const vector<Record>& records);
vector<string> getLabel(const vector<Record>& records);
vector<string> allLabels(const vector<string>& labels);
vector<vector<double> > normalize(const vector<string>& labels, const vector<string>& all_labels);
void read(char * filename, vector<vector<double> >& data, vector<vector<double> >& y);

#endif