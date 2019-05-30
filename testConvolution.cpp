#include "Convolution.hpp"

void test_kernel() {
	Kernel k(3, 3);
	k.print();
}

void test_conv() {
	Layers map = {
		{
			{1,2,3,4,5,6,7},
			{8,9,10,11,12,13,14},
			{15,16,17,18,19,10,21},
			{22,23,24,25,26,27,28},
			{29,30,31,32,33,34,35},
			{36,37,38,39,40,41,42},
			{43,44,45,46,47,48,49}
		},
		{
			{-1,2,3,4,5,6,7},
			{8,9,10,11,12,13,14},
			{15,16,17,18,19,10,21},
			{22,23,24,25,26,27,28},
			{29,30,31,32,33,34,35},
			{36,37,38,39,40,41,42},
			{43,44,45,46,47,48,0}
		}
	};
	Convolution convolution(map, 3, 3, 3, 1, 1);

	cout << "kernels:" << endl;
	convolution.print_kernels();

	convolution.conv(); // with relu
	// convolution.conv(false); // not with relu
	cout << "convolution map:" << endl;
	convolution.print_conv();

	convolution.max_pooling(2, 2);
	cout << "pooling map:" << endl;
	convolution.print_pooling();

	convolution.flatten();
	cout << "flatten vector:" << endl;
	convolution.print_flatten();
}

int main() {
	Kernel::setup(); // must appear just once in main function
	// test_kernel();
	test_conv();
	return 0;
}