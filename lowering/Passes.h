#ifndef PASSES_H
#define PASSES_H

#include "mlir/Pass/Pass.h"
#include "llvm/IR/LLVMContext.h"

#include <memory>

namespace choco {

    std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

    std::unique_ptr<mlir::Pass> createChocoToMLIRPass();

    mlir::ModuleOp lowerFromChocoToMlir(mlir::ModuleOp theModule,
                                        mlir::MLIRContext *mlirContext);


    std::unique_ptr<llvm::Module> lowerFromChocoToMLIRToLLVMIR(mlir::ModuleOp theModule, mlir::MLIRContext *mlirCtx, llvm::LLVMContext &llvmCtx);
}

#endif