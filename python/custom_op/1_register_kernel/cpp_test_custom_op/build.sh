
source ~/openvino/build/install/setupvars.sh 
mkdir -p build
cd build

cmake -DCMAKE_BUILD_TYPE=Debug ..
# cmake -DCMAKE_BUILD_TYPE=Release ..

make -j20

cd ..