#ifndef UTIL_H
#define UTIL_H

#include <vector>
#include <cmath>
#include <string>
using namespace std;

typedef vector<vector<vector<double> > > Layers;
typedef vector<vector<double> > Config;

void setup(); // put it once in main function for random (MUST DO !!!)
double sigmoid(double x);
vector<string> split(const string s, const string delim);
string& trim(string &s);
int argmax(vector<double> v);

#endif