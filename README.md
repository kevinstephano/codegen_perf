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

```
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=<Absolute Path to your Pytorch Root Directory> ..
cmake --build . --config Release
```

# Requirements

The profiling scripts require the Python package Pandas.

```
pip install -r requirements.txt
```

# Running Codegen Profiling
