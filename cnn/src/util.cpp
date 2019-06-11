#include "util.hpp"
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
using namespace std;

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

int argmax(vector<double> v) {
	double max = (v.size() > 0)? v[v.size()-1]: 0;
	int id = v.size();
	for (int i = v.size() - 1; i >= 0; i--) {
		if (max <= v[i]) {
			max = v[i];
			id = i;
		}
	}
	return id;
}
