NN: NN.cpp
	g++ --std=c++11 -o NN NN.cpp
bo: bo.cpp
	g++ -o bo bo.cpp
clean: 
	rm -rf NN bo *.o