set(LLVM_TARGET_DEFINITIONS ChocoOps.td)

mlir_tablegen(ChocoOps.h.inc -gen-op-decls)
mlir_tablegen(ChocoOps.cpp.inc -gen-op-defs)
mlir_tablegen(ChocoDialect.h.inc -gen-dialect-decls)
mlir_tablegen(ChocoDialect.cpp.inc -gen-dialect-defs)

add_public_tablegen_target(ChocoDialectIncGen)