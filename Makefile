NN: NN.cpp
	g++ --std=c++11 -o NN NN.cpp
testC: testConvolution.cpp Convolution.hpp
	g++ --std=c++11 -o testC testConvolution.cpp
bo: bo.cpp
	g++ -o bo bo.cpp
clean: 
	rm -rf NN bo *.o