add_library(ChocoLower
    ChocoLowering.cpp
    LowerToLLVM.cpp
    )


#include the directory containing ChocoLowering.h
target_include_directories(ChocoLower PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})