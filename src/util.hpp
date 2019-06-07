#ifndef UTIL_H
#define UTIL_H

#include <vector>
#include <cmath>
#include <string>
using namespace std;

typedef vector<vector<vector<double> > > Layers;

void setup(); // put it once in main function for random (MUST DO !!!)
double sigmoid(double x);
vector<string> split(const string s, const string delim);
string& trim(string &s);
int argmax(vector<double> v);
void read(char * filename, vector<vector<double> >& data, vector<vector<double> >& y);

#endif