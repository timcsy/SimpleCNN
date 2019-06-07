#ifndef RECORDS_H
#define RECORDS_H

#include <vector>
#include <string>
#include <fstream>
using namespace std;

struct Record {
	vector<double> data;
	string label;
	int id; // the corresponding index to the label_map
	vector<double> output;
	Record() {}
	Record(vector<double> data, string label, int id): data(data), label(label), id(id) {}
};

struct LabelMap {
	string label;
	vector<double> output;
	LabelMap() {}
	LabelMap(string label, vector<double> output): label(label), output(output) {}
};

class Records {
public:
	Records();
	Records(const string filename, const string delim, const string label, bool line_trim = true);
	Records(const string filename, const string delim, int col = 0, bool line_trim = true);
	vector<Record>& read_csv(const string filename, const string delim, int col = 0, bool line_trim = true);
	vector<Record>& read_csv(const string filename, const string delim, const string label, bool line_trim = true);
	vector<LabelMap>& read_label(const string filename);
	vector<LabelMap> getLabelMap() const { return label_map; }
	void setLabelMap(vector<LabelMap>& lm) { label_map = lm; normalize(); }
	void setLabelMap(Records& recs) { label_map = recs.getLabelMap(); normalize(); }
	void normalize();
	void shuffle();
	int size() { return records.size(); }
	Record operator[](int i) const;
private:
	vector<Record>& read_csv(const string filename, const string delim, int col, const string label, int header, bool line_trim);
	vector<Record> records;
	vector<LabelMap> label_map;
};

#endif