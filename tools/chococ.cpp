#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <memory>

#include "../codegen/AST.h"
#include "../codegen/lexer.cpp"
#include "../codegen/parser.cpp"
#include "../codegen/MLIRGen.cpp"
#include "../codegen/ASTPasses.h"
#include "../lowering/Passes.h"


#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"


#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LLVMRemarkStreamer.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/LTO/LTOBackend.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Pass.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h" // Include raw_ostream for raw_fd_ostream
#include "llvm/Support/FileSystem.h" 

using namespace choco;

void output_to_llfile(std::unique_ptr<llvm::Module> llvm_module)
{

    std::error_code ec;
    llvm::raw_fd_ostream file("example.ll", ec);
    
    if (ec) {
        llvm::errs() << "Error opening file: " << ec.message() << '\n';
        exit(0);
    }

    llvm_module->print(file, nullptr);

}


void dumpMLIR(Program& prg, int verify) {
    
    mlir::MLIRContext context;
    
    // Load our Dialect in this MLIR Context.
    context.getOrLoadDialect<mlir::choco::ChocoDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
    context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();

    std::unique_ptr<mlir::ModuleOp> mod = MLIRGenEntry(context, prg);

    if(verify == 1) {
        if(failed(mlir::verify(*mod))) {
            (*mod).emitError("Module couldn't be verified");
            exit(0);
        }
    }

    mod->dump();

    std::cout << "\n\n====================== Lowering started =====================\n" <<std::endl;    

    auto lower_module = choco::lowerFromChocoToMlir(*mod, &context);
    
    llvm::LLVMContext llvmCtx;
    
    auto emit_llvm_module = lowerFromChocoToMLIRToLLVMIR(*mod, &context, llvmCtx);

    std::cout<<"\n";
    std::cout<<"--------------------------- STD MLIR dump --------------------------\n"<<std::endl;

    lower_module.dump();
    
    std::cout<<"\n";
    std::cout<<"--------------------------- LLVMIR dump --------------------------\n"<<std::endl;
    emit_llvm_module->dump();

    // TODO: Output it to a .ll file, then make it run based on some flag
    if(verify == 1) {
        output_to_llfile(std::move(emit_llvm_module));
    }
    
    
}

int main(int argc, char* argv[])
{

    std::string code, filename;
    int verify;

    if(argc == 1)
    {
        std::cout << "Enter the .choco file to be compiled" << std::endl;
        exit(0);
    }

    else if(argc == 2)
    {
        // that means we got the .choco file
        // verify if the filename ends with .choco
        // set the filename variable
        filename = argv[1];

    }
    else if(argc == 3)
    {
        // that means we got the .choco file
        // verify if the filename ends with .choco
        // set the filename variable
        filename = argv[1];
        verify = (*argv[2]) - '0';
    } 
    else {
        std::cout<<"Too many arguments to the compiler"<<std::endl;
        exit(0);
    }

    std::ifstream inputFile(filename);

    if(inputFile) {
        std::ostringstream ss;
        ss<<inputFile.rdbuf(); // reading data
        code = ss.str();
    }


    Lexer l = Lexer(code);

    Parser p = Parser(l);

    std::shared_ptr<Program> ast = p.parseProgram();

    Program& prg = gatherPass(*ast);

    std::cout<<"============ Parsing errors =============== "<<std::endl;
    for(auto &e : p.getErrors()) {
        std::cout<<e<<std::endl;
    }
    std::cout<<std::endl;
    
    std::cout << ast->String() << std::endl;

    std::cout << "Printing complete" << std::endl;

    std::cout<<"--------------------------- MLIR dump --------------------------\n"<<std::endl;
    dumpMLIR(prg, verify);
    

    return -1;
}
