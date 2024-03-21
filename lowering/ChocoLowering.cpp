#include "include/choco/ChocoDialect.h"

#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"


#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LLVMRemarkStreamer.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/LTO/LTOBackend.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Pass.h"
#include <memory>


#include "Passes.h"

#include <iostream>

// Lowers the choco dialect ops to std MLIR dialect ops like arith, cf, func etc.



namespace lowering {

    class ConstOpLowering : public mlir::OpConversionPattern<mlir::choco::ConstOp> {
        public:
            using OpConversionPattern<mlir::choco::ConstOp>::OpConversionPattern;

            mlir::LogicalResult
            matchAndRewrite(mlir::choco::ConstOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const override {
                auto type = op.getType();
                
                mlir::TypedAttr value;
                auto v = op.getValue();
                value = rewriter.getIntegerAttr(type, v);
                rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op,type,value);
                return mlir::LogicalResult::success();
            }
    };

    class AddOpLowering : public mlir::OpConversionPattern<mlir::choco::AddOp>
    {
    public:
        using OpConversionPattern<mlir::choco::AddOp>::OpConversionPattern;

        mlir::LogicalResult
        matchAndRewrite(mlir::choco::AddOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const override
        {
            mlir::Type mlirType = getTypeConverter()->convertType(op.getType());
            rewriter.replaceOpWithNewOp<mlir::arith::AddIOp>(op, mlirType, adaptor.getLhs(), adaptor.getRhs());
            return mlir::LogicalResult::success();
        }
    };

    class MulOpLowering : public mlir::OpConversionPattern<mlir::choco::MulOp>
    {
    public:
        using OpConversionPattern<mlir::choco::MulOp>::OpConversionPattern;

        mlir::LogicalResult
        matchAndRewrite(mlir::choco::MulOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const override
        {
            mlir::Type mlirType = getTypeConverter()->convertType(op.getType());
            rewriter.replaceOpWithNewOp<mlir::arith::MulIOp>(op, mlirType, adaptor.getLhs(), adaptor.getRhs());
            return mlir::LogicalResult::success();
        }
    };


    // class AssignOpLowering : public mlir::OpConversionPattern<mlir::choco::AssignOp>
    // {
    // public:
    //     using OpConversionPattern<mlir::choco::AssignOp>::OpConversionPattern;

    //     mlir::LogicalResult
    //     matchAndRewrite(mlir::choco::AssignOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const override
    //     {

                // use LLVMload and stores and memref dialect 
    //         //TODO: How to lower assign to any existing dialect 
    //         mlir::Type mlirType = getTypeConverter()->convertType(op.getType());
    //         rewriter.replaceOpWithNewOp<mlir::arith::MulIOp>(op, mlirType, adaptor.getLhs(), adaptor.getRhs());
    //         return mlir::LogicalResult::success();
    //     }
    // };

    
} // namespace lowering

namespace choco
{
    struct ConvertChocoToMLIRPass
        : public mlir::PassWrapper<ConvertChocoToMLIRPass,
                                   mlir::OperationPass<mlir::ModuleOp>>
    {

        void getDependentDialects(mlir::DialectRegistry &registry) const override
        {
            registry.insert<mlir::BuiltinDialect, mlir::func::FuncDialect,
                            mlir::arith::ArithDialect, mlir::memref::MemRefDialect,
                            mlir::cf::ControlFlowDialect>();
        }

        void runOnOperation() final;

        virtual llvm::StringRef getArgument() const override { return "choco-to-mlir"; };
    };

    void ConvertChocoToMLIRPass::runOnOperation() {
        auto module = getOperation();
        mlir::TypeConverter typeConverter;
        mlir::RewritePatternSet patterns(&getContext());

        typeConverter.addConversion(
            [&](mlir::IntegerType type) -> mlir::Type { return type; });


        patterns.add<
            lowering::AddOpLowering,
            lowering::MulOpLowering,
            lowering::ConstOpLowering
            // lowering::AssignOpLowering
            >(typeConverter, &getContext());

        mlir::ConversionTarget target(getContext());

        target.addLegalOp<mlir::ModuleOp>();

        target.addLegalDialect<mlir::LLVM::LLVMDialect,
                               mlir::arith::ArithDialect,
                               mlir::func::FuncDialect,
                               mlir::memref::MemRefDialect,
                               mlir::cf::ControlFlowDialect>();
        
        target.addIllegalDialect<mlir::choco::ChocoDialect>();

        if (failed(applyPartialConversion(module, target, std::move(patterns))))
            signalPassFailure();
    
    }

    std::unique_ptr<mlir::Pass> createConvertChocoToMLIRPass() {
        return std::make_unique<ConvertChocoToMLIRPass>();
    }

    mlir::ModuleOp lowerFromChocoToMlir(mlir::ModuleOp theModule,
                                        mlir::MLIRContext *mlirContext) {
        
        mlir::PassManager pm(mlirContext);
        pm.addPass(createConvertChocoToMLIRPass());

        auto result = !mlir::failed(pm.run(theModule));
        if(!result)
            llvm::report_fatal_error(
                "The pass manager failed to lower choco to std MLIR dialect!");

        if (theModule.verify().failed())
            llvm::report_fatal_error("Verification of the final LLVMIR dialect failed!");

        return theModule;
    }

    // Function to run both the passes choco->std_MLIR->LLVM
    std::unique_ptr<llvm::Module> lowerFromChocoToMLIRToLLVMIR(mlir::ModuleOp theModule, 
                                                                mlir::MLIRContext *mlirCtx,
                                                                llvm::LLVMContext &llvmCtx) 
    {
        mlir::PassManager pm(mlirCtx);

        pm.addPass(createConvertChocoToMLIRPass());
        pm.addPass(createLowerToLLVMPass());
        pm.enableVerifier(false);

        // TODO: Passes throw error when returning from branches 
        auto res = pm.run(theModule);

        // TODO: find a way of dumping the module before verification
        theModule.dump();
        auto result = !mlir::failed(res);

        if (!result)
            llvm::report_fatal_error(
                "The pass manager failed lower Choco to LLVMIR dialect!");

        // Now that we ran all the lowering passes, verify the final output.
        if (theModule.verify().failed())
            llvm::report_fatal_error("Verification of the final LLVMIR dialect failed!");


        mlir::registerBuiltinDialectTranslation(*mlirCtx);
        mlir::registerLLVMDialectTranslation(*mlirCtx);

        auto llvmModule = mlir::translateModuleToLLVMIR(theModule, llvmCtx);

        if (!llvmModule)
            llvm::report_fatal_error("Lowering from LLVMIR dialect to llvm IR failed!");

        return llvmModule;
    }


}
