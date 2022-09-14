### JIT compile c++ code from python
This way is recommended for prototyping
```
python example_reduce_jit.py
```


### Compile and install c++ code from python
```
pip install -e .
python example_reduce.py
```

### Run rudimentary tests
The tests use the JIT compile solution
```
python tests.py
```

### Compile c++ code with cmake
```
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=<LIBTORCH_PATH> ..
./example_reduce
```
you might need to reference your cuda library
