Simple CNN
===

Using C++ to build a simple CNN Project that can recongnize digits.

Build
---
on UNIX:
```
make cmake
make build
```

Generally:
```
mkdir -p build
cd build
cmake ..
make
```

Run
---
on UNIX:
```
make run
```

Generally:
```
build/bin/scnn 'tmp/test_cnn_2.cnn' < 'data/example_input.txt'
```

Test
---
on UNIX:
```
make test
```

Generally:
```
build/bin/test
```

Clean
---
on UNIX:
```
make clean
```


Testing data
---
Please put at the test/data
Execute the program at the root folder
Put the output at tmp/

Image Format
---
```
784 of number
```