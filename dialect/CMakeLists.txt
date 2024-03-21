add_mlir_dialect_library(MLIRChoco
    ChocoDialect.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/choco

    DEPENDS
    ChocoDialectIncGen
)