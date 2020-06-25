# codegen_perf

# Build Directions

## Building Libtorch (if necessary)

`LibTorch` should be built when you build Pytorch from source. If it is not, you can elso execute.

```
cd <pytorch_root>

# Make a new folder to build in to avoid polluting the source directories
mkdir build_libtorch && cd build_libtorch

# You might need to export some required environment variables here.
Normally setup.py sets good default env variables, but you'll have to do
that manually.
python ../tools/build_libtorch.py
```

## Building Standalone App

Remember to delete your old `build` diectory, first.
```
./run_build.sh <Path to your Pytorch Root Directory>
```

```
mkdir build
cd build
cmake -DCMAKE_CXX_FLAGS="-I <Absolute Path to your Pytorch Root Directory>" -DCMAKE_PREFIX_PATH=<Absolute Path to your Pytorch Root Directory> ..
cmake --build . --config Release
```

# Requirements

The profiling scripts require the Python package Pandas.

```
pip install -r requirements.txt
```

# Running Codegen Profiling

Running profiling requires the use of sudo as nvprof requires root permissions for security.  This can override your python path if you run from a virtual environment.

### Example
```
sudo <path to python>/python prof_codegen.py --block-x-start 10 --block-x-stop 320 --block-x-inc 10 --thread-x-start 128 --thread-x-stop 128 --thread-x-inc 128
```
