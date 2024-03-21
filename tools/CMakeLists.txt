get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set (LIBS
    ${LLVM_LIBRARIES}
    ${dialect_libs}
    ${conversion_libs}
    codegen
    MLIRChoco
    ChocoLower
    )

# Add compiler options to enable RTTI

add_llvm_executable(chococ chococ.cpp)


# enable RTTI for the target
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(chococ PRIVATE "-frtti")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(chococ PRIVATE "/GR")
endif()

llvm_update_compile_flags(chococ)
target_link_libraries(chococ PRIVATE ${LIBS})

mlir_check_all_link_libraries(chococ)