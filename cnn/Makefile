all: cmake make

build: FORCE
	cd build; make
cmake: FORCE
	mkdir -p build; cd build; cmake ..
run: FORCE
	@build/bin/scnn
test: FORCE
	@build/bin/test
clean: 
	rm -rf build
FORCE:
