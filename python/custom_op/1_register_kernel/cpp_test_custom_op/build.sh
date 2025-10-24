
source ~/openvino/build/install/setupvars.sh 
mkdir -p build
cd build
cmake ..

make -j20

cd ..