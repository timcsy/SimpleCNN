NN: NN.cpp
	g++ --std=c++11 -o NN NN.cpp
testC: testConvolution.cpp Convolution.hpp
	g++ --std=c++11 -o testC testConvolution.cpp
testBS: testBinaryStream.cpp BinaryStream.hpp
	g++ --std=c++11 -o testBS testBinaryStream.cpp
clean: 
	rm -rf NN testC testBS *.o *output.txt