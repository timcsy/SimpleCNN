#ifndef UTIL_H
#define UTIL_H

#include <vector>
#include <cmath>
using namespace std;

typedef vector<vector<vector<double> > > Layers;

double sigmoid(double x);
void read(char * filename, vector<vector<double> >& data, vector<vector<double> >& y);

#endif