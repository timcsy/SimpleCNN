#include "util.hpp"
using namespace std;

double sigmoid(double x) {
	return (1 / (1 + exp(-x)));
}