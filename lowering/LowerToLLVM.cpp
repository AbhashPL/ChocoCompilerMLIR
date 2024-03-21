#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/DialectConversion.h"



#include "Passes.h"
#include "include/choco/ChocoDialect.h"

namespace choco {

    struct LowerToLLVMPass
        : public mlir::PassWrapper<LowerToLLVMPass, 
                                    mlir::OperationPass<mlir::ModuleOp>> {

        void getDependentDialects(mlir::DialectRegistry &registry) const override {
            registry.insert<mlir::LLVM::LLVMDialect>();
        }
        
        void runOnOperation() final;

        virtual llvm::StringRef getArgument() const override { return "choco-mlir-to-llvm"; }
    
    };


    void LowerToLLVMPass::runOnOperation() {
        mlir::LLVMConversionTarget target(getContext());
        target.addLegalOp<mlir::ModuleOp>();

        mlir::LLVMTypeConverter typeConverter(&getContext());

        mlir::RewritePatternSet patterns(&getContext());
        
        mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
        mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
        populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
        populateFuncToLLVMConversionPatterns(typeConverter, patterns);

        auto module = getOperation();

        if (failed(applyFullConversion(module, target, std::move(patterns))))
            signalPassFailure();
        
    }

    std::unique_ptr<mlir::Pass> createLowerToLLVMPass() {
        return std::make_unique<LowerToLLVMPass>();
    }

}