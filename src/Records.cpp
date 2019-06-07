#include "Records.hpp"
#include "util.hpp"
#include <algorithm> // std::random_shuffle
#include <random>       // std::default_random_engine
#include <chrono>       // std::chrono::system_clock
using namespace std;

Records::Records(const string filename, const string delim, int col, bool line_trim) {
	read_csv(filename, delim, col, line_trim);
}

Records::Records(const string filename, const string delim, const string label, bool line_trim) {
	read_csv(filename, delim, label, line_trim);
}

vector<Record>& Records::read_csv(const string filename, const string delim, int col, const string label, int header, bool line_trim) {
	fstream fin;
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

vector<Record>& Records::read_csv(const string filename, const string delim, int col, bool line_trim) {
	return read_csv(filename, delim, col, "", false, line_trim);
}

vector<Record>& Records::read_csv(const string filename, const string delim, const string label, bool line_trim) {
	return read_csv(filename, delim, 0, label, true, line_trim);
}

vector<LabelMap>& Records::read_label(const string filename) {
	vector<string> labels;
	fstream fin(filename.c_str(), ios::in);
	string s;
	while (getline(fin, s)) {
		trim(s);
		bool is_found = false;
		for (int i = 0; i < labels.size(); i++) {
			if (s == labels[i]) {
				is_found = true;
				break;
			}
		}
		if (!is_found) labels.push_back(s);
	}
	fin.close();

	for (int i = 0; i < labels.size(); i++) {
		LabelMap lm;
		lm.label = labels[i];
		for (int j = 0; j < labels.size(); j++) {
			if (i == j) lm.output.push_back(1);
			else lm.output.push_back(0);
		}
		label_map.push_back(lm);
	}

	normalize();
	return label_map;
}

void Records::normalize() {
	for (int i = 0; i < records.size(); i++) {
		for (int j = 0; j < label_map.size(); j++) {
			if (records[i].label == label_map[j].label) {
				records[i].id = j;
				break;
			}
		}
	}
}

void Records::shuffle() {
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::shuffle(records.begin(), records.end(), std::default_random_engine(seed));
}

Record Records::operator[](int i) const {
	if (i >= records.size()) throw "Records index out of range";
	Record rec = records[i];
	rec.output = label_map[rec.id].output;
	return rec;
}