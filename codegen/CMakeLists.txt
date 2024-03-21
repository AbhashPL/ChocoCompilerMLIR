#Add source files to the library
add_library(codegen
    parser.cpp
    lexer.cpp
    MLIRGen.cpp
    ASTPasses.cpp
    )


#include the directory containing AST.h
target_include_directories(codegen PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})