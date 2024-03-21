#include "include/choco/ChocoDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
// #include "mlir/Transforms/InliningUtils.h"
#include "mlir/Interfaces/FunctionImplementation.h"

using namespace mlir;
using namespace mlir::choco;

#include "include/choco/ChocoDialect.cpp.inc"

void ChocoDialect::initialize()
{

    // register the operations with the choco dialect
    addOperations<
        #define GET_OP_LIST
        #include "include/choco/ChocoOps.cpp.inc"
        >();
}


void ConstOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, int value) {
    auto inttype = builder.getI32Type();
    //builder.getIntegerType(32, false);
    auto dataAttr = IntegerAttr::get(inttype, value);
    ConstOp::build(builder, state, inttype, dataAttr);            
}


void FuncOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   std::string name, mlir::FunctionType type,
                   std::vector<mlir::NamedAttribute> attrs) {
    
    // FunctionOpInterface provides a convenient `build` method that will populate
    // the state of our FuncOp, and create an entry block.

    // this function comes from functionInterface.td file
    buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}





// Builder for a ReturnOp with no operands
void ReturnOp::build(mlir::OpBuilder &builder, mlir::OperationState &state) {
    ReturnOp::build(builder, state, std::nullopt);
}

void CallOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, std::string callee, std::vector<mlir::Value> args) {
    state.addTypes(builder.getI32Type());
    state.addOperands(args);
    state.addAttribute("callee",
                        mlir::SymbolRefAttr::get(builder.getContext(), callee));
}

#define GET_OP_CLASSES
#include "include/choco/ChocoOps.cpp.inc"
