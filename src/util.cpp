#include "util.hpp"
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
using namespace std;

void print(vector<vector<double> > v) {
	for (int i = 0; i < v.size(); i++) {
		for (int j = 0; j < v[i].size(); j++) {
			cout << v[i][j] << " ";
		}
		cout << endl;
	}
}

void setup() { // put it once in main function for random (MUST DO !!!)
	srand((unsigned)time(NULL));
}

double sigmoid(double x) {
	return (1 / (1 + exp(-x)));
}

vector<string> split(const string s, const string delim) {
	vector<string> v;
  string::size_type pos1, pos2;
  pos2 = s.find(delim);
  pos1 = 0;
  while (string::npos != pos2) {
    v.push_back(s.substr(pos1, pos2 - pos1));
    pos1 = pos2 + delim.size();
    pos2 = s.find(delim, pos1);
  }
  if(pos1 != s.length())
    v.push_back(s.substr(pos1));
	return v;
}

string& trim(string &s) {
	if (s.empty()) return s;
	s.erase(0,s.find_first_not_of(" "));
	s.erase(s.find_last_not_of(" ") + 1);
	return s;
}

vector<Record> read_csv(const string filename, const string delim, int col, const string label, int header, bool line_trim) {
	fstream fin;
	vector<Record> records;

	string line;
	fin.open(filename.c_str(), ios::in);
	if (header) {
		getline(fin, line);
		if (line_trim) trim(line);
		vector<string> row = split(line, delim);
		if (label != "") for (int i = 0; i < row.size(); i++) if (row[i] == label) { col = i; break; }
	}
	while (getline(fin, line)) {
		Record rec;
		if (line_trim) trim(line);
		vector<string> row = split(line, delim);
		for (int i = 0; i < row.size(); i++) {
			if (i == col) rec.label = row[i];
			else rec.data.push_back(stod(row[i]));
		}
		records.push_back(rec);
	}
	fin.close();

	return records;
}

vector<Record> read_csv(const string filename, const string delim, int col, bool line_trim) {
	return read_csv(filename, delim, col, "", false, line_trim);
}

vector<Record> read_csv(const string filename, const string delim, const string label, bool line_trim) {
	return read_csv(filename, delim, 0, label, true, line_trim);
}

vector<vector<double> > getData(const vector<Record>& records) {
	vector<vector<double> > data;
	for (int i = 0; i < records.size(); i++) data.push_back(records[i].data);
	return data;
}

vector<string> getLabel(const vector<Record>& records) {
	vector<string> labels;
	for (int i = 0; i < records.size(); i++) labels.push_back(records[i].label);
	return labels;
}

vector<string> allLabels(const vector<string>& labels) {
	vector<string> label_set;
	for (int i = 0; i < labels.size(); i++) {
		bool is_found = false;
		for (int j = 0; j < label_set.size(); j++) {
			if (labels[i] == label_set[j]) {
				is_found = true;
				break;
			}
		}
		if (!is_found) label_set.push_back(labels[i]);
	}
	return label_set;
}

vector<vector<double> > normalize(const vector<string>& labels, const vector<string>& all_labels) {
	vector<vector<double> > samples;
	vector<vector<double> > alias;
	// create vector table
	for (int i = 0; i < all_labels.size(); i++) {
		vector<double> v;
		for (int j = 0; j < all_labels.size(); j++) {
			if (i == j) v.push_back(1);
			else v.push_back(0);
		}
		alias.push_back(v);
	}

	for (int i = 0; i < labels.size(); i++) {
		for (int j = 0; j < all_labels.size(); j++) {
			if (labels[i] == all_labels[j]) {
				samples.push_back(alias[j]);
				break;
			}
		}
	}
	return samples;
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