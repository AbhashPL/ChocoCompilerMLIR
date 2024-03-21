#ifndef DIALECT_CHOCO_CHOCODALECT_H
#define DIALECT_CHOCO_CHOCODALECT_H

// Required because the .h.inc file refers to MLIR classes and does not itself
//  have any includes.
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Dialect.h"

// Include the tabled-gen'd header file containing the declaration of the toy dialect
#include "include/choco/ChocoDialect.h.inc"

#define GET_OP_CLASSES
#include "include/choco/ChocoOps.h.inc"

#endif // DIALECT_CHOCO_CHOCODALECT_H