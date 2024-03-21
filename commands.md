
cmake -G Ninja .. -DMLIR_DIR=$MLIR_PREFIX/lib/cmake/mlir -DLLVM_DIR=$MLIR_PREFIX/lib/cmake/llvm -DCMAKE_BUILD_TYPE=Debug -DLLVM_ENABLE_LLD=ON


# For generating the dialect include files
    -> cmake --build ./build --target ChocoDialectIncGen
    -> cmake --build ./build --target chococ