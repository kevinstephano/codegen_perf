
mkdir build
cd build
cmake -DCMAKE_CXX_FLAGS="-I $1 -I $1/aten/src/" -DCMAKE_PREFIX_PATH=$1 ..
cmake --build . --config Release
