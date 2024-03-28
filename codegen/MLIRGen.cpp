// Generate .mlir files from choco source code. One operation at a time

#include "include/choco/ChocoDialect.h"
#include "AST.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Operation.h"


#include <memory>

using namespace mlir::choco;
using namespace choco;


class MLIRGen {
    public:
        MLIRGen(mlir::MLIRContext &context) 
            : builder(&context) {
                
                typeEnv["print"] = nullptr; // return type of print is null

            } // the builder is intialized with the address of the context object
        
        mlir::ModuleOp mlirGen(Program& prg) {
            
            theModule = mlir::ModuleOp::create(builder.getUnknownLoc(), "my_module");
            builder.setInsertionPointToEnd(theModule.getBody());
            
            // Add declaration for the print function in the runtime
            std::vector<mlir::Type> print_args{builder.getI32Type()};

            auto funcType = builder.getFunctionType(print_args, std::nullopt);

            auto printFunc = mlir::func::FuncOp::create(builder.getUnknownLoc(), "print", funcType);
            printFunc.setVisibility(mlir::SymbolTable::Visibility::Private);

            theModule.push_back(printFunc);

            builder.setInsertionPointToEnd(theModule.getBody());

            for(auto &st : prg.getStatements()) {
                
                switch(st->getKind()) {
                    case ExprStmt :
                        {
                            std::shared_ptr<ExpressionStatement> es = std::dynamic_pointer_cast<ExpressionStatement>(st);

                            if (!es)
                                return nullptr;

                            mlirGen(*es);
                            break;
                        }

                    case VarDeclStmt :
                        {
                            std::shared_ptr<VarDeclStatement> vs = std::dynamic_pointer_cast<VarDeclStatement>(st);

                            if(!vs)
                                return nullptr;
                            
                            mlirGen(*vs);
                            break;
                        }

                    case AssignStmt :
                        {
                            std::shared_ptr<AssignStatement> as = std::dynamic_pointer_cast<AssignStatement>(st);

                            if(!as)
                                return nullptr;
                            
                            mlirGen(*as);
                            break;
                        }

                    case FuncStmt:
                        {
                            std::shared_ptr<Function> fs = std::dynamic_pointer_cast<Function>(st);
                         
                            if(!fs)
                                return nullptr;
                            
                            
                            mlirGen(*fs);
                            break;
                        }

                    case RetStmt:
                        break;

                    case IfStmt:
                        {
                            std::shared_ptr<IfStatement> is = std::dynamic_pointer_cast<IfStatement>(st);
                         
                            if(!is)
                                return nullptr;
                            
                            mlirGen(*is);
                            break;
                        }

                }
            }

            return theModule;
        }

        void declare(std::string s, mlir::Value v) {
            symblTbl[s] = v;
        }

        void getType(std::vector<std::shared_ptr<Identifier>> &params, std::vector<mlir::Type> &vec) {
            // for now all the args are integers
            for(int i = 0; i<params.size(); i++) {
                vec.push_back(builder.getI32Type());
            }
        }

        // return the name of the last op in the block
        std::string getLastOpName(mlir::Block& blk) {
            auto &op = blk.back();
            auto opName = op.getName().getStringRef().str();
            return opName;
        }

        bool check_if_all_are_null(std::vector<int> &vals)
        {
            bool flag = true;

            for(auto e : vals) {
                if(e == 0) {
                    continue;
                } else {
                    flag = false;
                    break;
                }
            }

            return flag;
        }

        bool check_if_all_are_not_null(std::vector<int> &vals)
        {
            bool flag = true;
            
            for(auto e : vals) {
                if(e == 1) {
                    continue;
                } else {
                    flag = false;
                    break;
                }
            }
            
            return flag;
        }

        // TODO:
        // create the function type based on the return,input types and then add it to map
        // Check if the stuff being returned in a function is all of the same type
        // if not error out from this function
        mlir::FunctionType createFuncType(std::string funcName, std::vector<mlir::Type>& argTypes, 
            std::unordered_map<std::shared_ptr<ReturnStatement>, int>& retMap) 
        {
            
            std::vector<int> vals;
            vals.reserve(retMap.size());

            for (auto kv : retMap)
            {
                vals.push_back(kv.second);
            }
           
            mlir::FunctionType funcType;

            if(check_if_all_are_null(vals)) {
                
                funcType = builder.getFunctionType(argTypes, std::nullopt);
                // register the function type in the type envionment
                typeEnv[funcName] = nullptr;

            } else if(check_if_all_are_not_null(vals)) {
                
                funcType = builder.getFunctionType(argTypes, builder.getI32Type());
                // register the function type in the type envionment
                typeEnv[funcName] = builder.getI32Type();

            } else {
                std::cout << "All return statements should return the value of the same type " << std::endl;
                exit(0);
            }

            return funcType;
        }

        mlir::Type getFuncType(std::string s) {
            return typeEnv[s];
        }

        
        mlir::func::FuncOp mlirGen(Function& fs) {

            // use the FuncOp that is builtin MLIR and emit that, instead of your own choco::FuncOp
            auto location = builder.getUnknownLoc();
            int num_params = fs.getParams().size();
            std::vector<mlir::Type> argTypes;
            std::vector<std::shared_ptr<Identifier>> &params = fs.getParams();
            getType(params, argTypes); // this will fill the vec with types
            
            fs.gatherReturns(fs.getBody()); // this will collect the mapping of returns

            auto &retMap = fs.getReturnsMap();

            mlir::FunctionType funcType = createFuncType(fs.getName(), argTypes, retMap);
            
            auto func = builder.create<mlir::func::FuncOp>(location, fs.getName(), funcType);

            auto entry = func.addEntryBlock();
            
            // declare all the params in the symblTbl
            auto blockArguments = entry->getArguments();

            auto nameValue = llvm::zip(params, blockArguments);
            for (auto nv : nameValue)
            {
                declare(std::get<0>(nv)->getName(),
                        std::get<1>(nv));
            }

            builder.setInsertionPointToStart(entry);

            std::shared_ptr<BlockStatement> body = fs.getBody();

            // delete code after an if..else stmt, if the then and else branches both have returns in them
            // This code could to moved to ASTPasses.h
            auto &original_stmts = body->getStatements();
            std::vector<std::shared_ptr<Statement>>::iterator it = original_stmts.begin();

            while (it!= original_stmts.end())
            {
                auto st = *it;
                if(st->getKind() == IfStmt) {
                    // clear eveything after "it" and break
                    // dyn_cast to if
                    std::shared_ptr<IfStatement> is = std::dynamic_pointer_cast<IfStatement>(st);
                    
                    if(is->getEls()) {
                        
                        if(is->hasReturnInBothBranches()) {
                            ++it;
                            original_stmts.erase(it, original_stmts.end());
                            break;
                        } else {
                            ++it;
                            continue;
                        }

                    } else {
                        ++it;
                        continue;
                    }

                } else {
                    ++it;
                }
            }

            // codegen the body of the function
            if (mlirGen(*body) == false)
            {
                std::cout<<"The function body could not be codegen'd for : "<<fs.getName()<<" "<<std::endl;
                func.erase();
                return nullptr;
            }

            auto blks = func.getBlocks().size();
            std::cout<<"Num of basic blocks are :"<<blks<<std::endl;

            
            builder.setInsertionPointAfter(func); // so that we codegen after this funcOperation

            return func;
        }

        void mlirGen(IfStatement &is)
        {
            auto location = builder.getUnknownLoc();
            auto els  = is.getEls();
            auto cond = is.getCond();
            mlir::Value condVal =  mlirGen(cond);
            auto parent_region = condVal.getParentRegion();
            auto ifInsertPoint = builder.saveInsertionPoint();

            if(!els) {
                    // No else block so we are cond_branching to then and merge blocks respectively

                    mlir::Block *mergeBlock = builder.createBlock(parent_region);

                    
                    mlir::Block *thenBlock = builder.createBlock(parent_region);
                    builder.setInsertionPointToStart(thenBlock);
                    auto then = is.getThen();
                    mlirGen(*then);

                    if (getLastOpName(*thenBlock) != "func.return")
                    {
                        builder.setInsertionPointToEnd(thenBlock);
                        builder.create<mlir::cf::BranchOp>(builder.getUnknownLoc(), mergeBlock);
                    }

                    builder.restoreInsertionPoint(ifInsertPoint);
                    builder.create<mlir::cf::CondBranchOp>(location, condVal, thenBlock, mergeBlock);
                    builder.setInsertionPointToEnd(mergeBlock);
                } 
                else {

                    if (is.hasReturnInBothBranches())
                    {
                        // merge block not needed at all

                        mlir::Block *thenBlock = builder.createBlock(parent_region);
                        builder.setInsertionPointToStart(thenBlock);
                        auto then = is.getThen();
                        mlirGen(*then);

                        if (getLastOpName(*thenBlock) != "func.return")
                        {
                            builder.setInsertionPointToEnd(thenBlock);
                            //builder.create<mlir::cf::BranchOp>(builder.getUnknownLoc(), mergeBlock);
                        }

                        mlir::Block *elseBlock = builder.createBlock(parent_region);
                        builder.setInsertionPointToStart(elseBlock);
                        mlirGen(*els);

                        if (getLastOpName(*elseBlock) != "func.return")
                        {
                            builder.setInsertionPointToEnd(elseBlock);
                            //builder.create<mlir::cf::BranchOp>(builder.getUnknownLoc(), mergeBlock);
                        }

                        builder.restoreInsertionPoint(ifInsertPoint);
                        auto condBrOp = builder.create<mlir::cf::CondBranchOp>(location, condVal, thenBlock, elseBlock);
                    }

                    else if (is.hasReturnInOneBranch())
                    {
                        // has return either in thn or els branch
                        // then we will need merge block
                        mlir::Block *mergeBlock = builder.createBlock(parent_region);

                        mlir::Block *thenBlock = builder.createBlock(parent_region);
                        builder.setInsertionPointToStart(thenBlock);
                        auto then = is.getThen();
                        mlirGen(*then);

                        if (getLastOpName(*thenBlock) != "func.return")
                        {
                            builder.setInsertionPointToEnd(thenBlock);
                            builder.create<mlir::cf::BranchOp>(builder.getUnknownLoc(), mergeBlock);
                        }

                        mlir::Block *elseBlock = builder.createBlock(parent_region);
                        builder.setInsertionPointToStart(elseBlock);
                        mlirGen(*els);

                        if (getLastOpName(*elseBlock) != "func.return")
                        {
                            builder.setInsertionPointToEnd(elseBlock);
                            builder.create<mlir::cf::BranchOp>(builder.getUnknownLoc(), mergeBlock);
                        }

                        builder.restoreInsertionPoint(ifInsertPoint);
                        auto condBrOp = builder.create<mlir::cf::CondBranchOp>(location, condVal, thenBlock, elseBlock);
                        
                        builder.setInsertionPointToEnd(mergeBlock);
                    }
                    else
                    {
                        // Has return in none branches
                        // then we will need merge block

                        mlir::Block *mergeBlock = builder.createBlock(parent_region);

                        mlir::Block *thenBlock = builder.createBlock(parent_region);
                        builder.setInsertionPointToStart(thenBlock);
                        auto then = is.getThen();
                        mlirGen(*then);

                        if (getLastOpName(*thenBlock) != "func.return")
                        {
                            builder.setInsertionPointToEnd(thenBlock);
                            builder.create<mlir::cf::BranchOp>(builder.getUnknownLoc(), mergeBlock);
                        }

                        mlir::Block *elseBlock = builder.createBlock(parent_region);
                        builder.setInsertionPointToStart(elseBlock);
                        mlirGen(*els);

                        if (getLastOpName(*elseBlock) != "func.return")
                        {
                            builder.setInsertionPointToEnd(elseBlock);
                            builder.create<mlir::cf::BranchOp>(builder.getUnknownLoc(), mergeBlock);
                        }

                        builder.restoreInsertionPoint(ifInsertPoint);
                        auto condBrOp = builder.create<mlir::cf::CondBranchOp>(location, condVal, thenBlock, elseBlock);

                        builder.setInsertionPointToEnd(mergeBlock);
                    }
            }
        }

        
        // custom FuncOp for the choco dialect
        // it has some type mismatch error
        mlir::choco::FuncOp mlirGen(Function& fs, bool flag) {
            
            int num_params = fs.getParams().size();
            std::vector<mlir::Type> argTypes;
            std::vector<std::shared_ptr<Identifier>> &params = fs.getParams();
            getType(params, argTypes); // this will fill the vec with types

            
            builder.setInsertionPointToEnd(theModule.getBody()); // The function will now be attached at the end            

            mlir::FunctionType funcType;
            
            if(fs.hasReturn()){
                // UI32 return type
                funcType = builder.getFunctionType(argTypes, builder.getI32Type());
            }
            else {
                // no return type
                funcType = builder.getFunctionType(argTypes, std::nullopt);
            }

            // myFunction is now a region
            auto myFunction = builder.create<mlir::choco::FuncOp>(builder.getUnknownLoc(), fs.getName(), funcType);

            if(!myFunction)
                return nullptr;

            // Now we have a basic block auto created by the above builder.create call...
            mlir::Block &entryBlock = myFunction.front(); // gives us the entry block in the region

            // declare all the params in the symblTbl
            auto blockArguments = entryBlock.getArguments();
            
            auto nameValue = llvm::zip(params, blockArguments);
            for(auto nv : nameValue) {
                declare(std::get<0>(nv)->getName(),
                std::get<1>(nv));
            }

            // Set the insertion point in the builder to the beginning of the function
            // body, it will be used throughout the codegen to create operations in this
            // function.
            builder.setInsertionPointToStart(&entryBlock);

            std::shared_ptr<BlockStatement> body = fs.getBody();

            
            // codegen the body
            if(mlirGen(*body) == false) {
                myFunction.erase();
                return nullptr;
            }


            // Add a return statement to terminate the basic block, it's the only terminator we have right now
            // there is an option of building the return op even if the op has nothing to return.
            
            // check if there's a return op at the end of the entryBlock, if not create one with no arguments 
            // using that custom build function that you defined

            auto &op = entryBlock.back();
            auto opName = op.getName().getStringRef().str();

            if(opName == "choco.return") {
                // that means the already exists a ReturnOp at the end
                myFunction.setType(builder.getFunctionType(myFunction.getFunctionType().getInputs(), builder.getI32Type()));
            } else {
                builder.create<ReturnOp>(builder.getUnknownLoc());
            }
            
            return myFunction;
        }

        void mlirGen(ReturnStatement& rs) {
            mlir::Location loc = builder.getUnknownLoc();
            std::shared_ptr<Expression> rexp = rs.getReturnValue();

            if(rexp) {
                mlir::Value val = mlirGen(rexp);
                std::vector<mlir::Value> rvec{val};

                auto retOp = builder.create<mlir::func::ReturnOp>(loc, rvec);

                std::cout << " parent of returnOp = " << retOp.getParentOp()->getName().getStringRef().str() << std::endl
                          << std::endl;

            } else {
                auto retOp = builder.create<mlir::func::ReturnOp>(loc, std::nullopt);

                std::cout << " parent of returnOp = " << retOp.getParentOp()->getName().getStringRef().str() << std::endl
                          << std::endl;
            }
            
        }

        bool mlirGen(BlockStatement& bs) {
            
            std::vector<std::shared_ptr<Statement>> sts = bs.getStatements();

            for(auto st : sts) {

                switch(st->getKind()) {
                    case ExprStmt :
                        {
                            std::shared_ptr<ExpressionStatement> es = std::dynamic_pointer_cast<ExpressionStatement>(st);

                            if (!es)
                                return false;

                            mlirGen(*es);
                            break;
                        }

                    case VarDeclStmt :
                        {
                            std::shared_ptr<VarDeclStatement> vs = std::dynamic_pointer_cast<VarDeclStatement>(st);

                            if(!vs)
                                return false;
                            
                            mlirGen(*vs);
                            break;
                        }

                    case AssignStmt :
                        {
                            std::shared_ptr<AssignStatement> as = std::dynamic_pointer_cast<AssignStatement>(st);

                            if(!as)
                                return false;
                            
                            mlirGen(*as);
                            break;
                        }

                    case RetStmt:
                        {
                            std::shared_ptr<ReturnStatement> rs = std::dynamic_pointer_cast<ReturnStatement>(st);

                            if(!rs)
                                return false;

                            mlirGen(*rs);
                            break;
                        }

                    case IfStmt:
                    {   
                        std::shared_ptr<IfStatement> is = std::dynamic_pointer_cast<IfStatement>(st);
                        if(!is)
                            return false;

                        mlirGen(*is);

                        break;
                    }
                        
                }
            }

            return true;
        }

        mlir::Value mlirGen(ExpressionStatement& expStmt)
        {
            return mlirGen(expStmt.getExp());
        }

        void mlirGen(VarDeclStatement& varStmt) {
            
            mlir::Value val = mlirGen(varStmt.getValueExpr()); // codegen the RHS
            std::string name = varStmt.getIdent()->getName();
            
            // TODO: var_decl to llvm


            // declare the variable in environment
            declare(name,val);
        }

        void mlirGen(AssignStatement& asStmt) {
            auto location = builder.getUnknownLoc();

            mlir::Value val = mlirGen(asStmt.getRight());
            std::string name = asStmt.getIdent()->getName();

            if(symblTbl.count(name) == 0)
            {
                std::cout<<"Assign before definition error!!"<<std::endl;
                return;
            }

            builder.create<mlir::choco::AssignOp>(location, symblTbl[name], val);
            
            // // put in new bindings in the symblTbl
            // symblTbl.insert_or_assign(name, val);

        }


        mlir::Value mlirGen(IntegerLiteral& num) {
            
            mlir::Location loc = builder.getUnknownLoc();
            
            uint32_t value = num.getVal();
            
            mlir::Value v = builder.create<ConstOp>(loc, value);
            builder.setInsertionPointAfterValue(v);
            return v;
        }


        mlir::Value mlirGen(InfixExpression& inExp) {
            // parse LHS, parse RHS

            std::shared_ptr<Expression> left = inExp.getLeft();
            mlir::Value lhs = mlirGen(left);
            
            if(!lhs)
                return nullptr;

            std::shared_ptr<Expression> right = inExp.getRight();
            mlir::Value rhs = mlirGen(right);

            if(!rhs)
                return nullptr;


            std::string ope = inExp.getOp();
            char op = ope.at(0);

            switch(op) {
                case '+' : 
                {
                    auto inttype = builder.getI32Type();
                    return builder.create<AddOp>(builder.getUnknownLoc(), inttype, lhs, rhs);
                }

                case '*' :
                {
                    auto inttype = builder.getI32Type();
                    return builder.create<MulOp>(builder.getUnknownLoc(), inttype, lhs, rhs);
                }

                case '<':
                {   
                    return builder.create<mlir::arith::CmpIOp>(builder.getUnknownLoc(), mlir::arith::CmpIPredicate::ult, lhs, rhs);
                }

                case '>':
                {
                    return builder.create<mlir::arith::CmpIOp>(builder.getUnknownLoc(), mlir::arith::CmpIPredicate::ugt, lhs, rhs);
                }
                    // add more binary opeartors here

            }

            std::cout << "No such binary operator" << std::endl;
            return nullptr;
        }
        
        mlir::Value mlirGen(CallExpression& cExp) {
            
            std::string callee = cExp.getCallee();
            //TODO: check if the following function is present for calling



            std::vector<mlir::Value> operands;

            for(auto &expr : cExp.getArgs()) {
                auto arg = mlirGen(expr);

                if(!arg)
                    return nullptr;

                operands.push_back(arg);
            }

            //TODO: if the function being called is a 'print', then make sure that it has only one operand and that too of type int32
            if(callee == "print") {
                if(operands.size() > 1) {
                    std::cout<<"print takes in only one argument if type integer"<<std::endl;
                    exit(0);
                }
            }

            auto ret_type = getFuncType(callee);

            mlir::func::CallOp callop;

            if(ret_type) {
                callop = builder.create<mlir::func::CallOp>(builder.getUnknownLoc(), ret_type, callee, operands);
                return callop.getResult(0);
            } else {
                callop = builder.create<mlir::func::CallOp>(builder.getUnknownLoc(), std::nullopt, callee, operands);
                return nullptr;
            }

            
        }

        mlir::Value mlirGen(std::shared_ptr<Expression> exp) {

            // This function is responsible for dispatching the correct mlirGen function based on what kind of expression it is
            switch (exp->getKind())
            {
            case IntegerLiteralExp :
                {
                    std::shared_ptr<IntegerLiteral> inLit = std::dynamic_pointer_cast<IntegerLiteral>(exp);
                    return mlirGen(*inLit);
                }

            case IdentifierExp :
                {
                    std::shared_ptr<Identifier> ident = std::dynamic_pointer_cast<Identifier>(exp);
                    std::string name = ident->getName();

                    // check if the identifier is in current scope, if not error out
                    if(symblTbl.count(name) == 0) {
                        // doesn't exist in symblTBL
                        std::cout << "Use before definition error !" << std::endl;
                        return nullptr;
                    }
                    else
                        return symblTbl[name];
                }

            case InfixExpressionExp :
                {
                    std::shared_ptr<InfixExpression> inExp = std::dynamic_pointer_cast<InfixExpression>(exp);
                    return mlirGen(*inExp);
                }

            case CallExp :
                {   
                    std::shared_ptr<CallExpression> cExp = std::dynamic_pointer_cast<CallExpression>(exp);
                    return mlirGen(*cExp);
                }

            }

            std::cout << "No such expression type" <<std::endl;
            return nullptr;
        }



    private:
        /// A "module" matches a source file
        mlir::ModuleOp theModule;

        /// The builder is a helper class to create IR inside a function, the builder is stateful
        /// It keeps an "insertion point", i.e. the place where next operations will be inserted
        mlir::OpBuilder builder;

        /// The symbol table maps a variable name to a value in the current scope.
        std::unordered_map<std::string, mlir::Value> symblTbl;

        std::unordered_map<std::string, mlir::Type> typeEnv;

        std::vector<mlir::OperandRange> OperandVec;
};


namespace choco {
    std::unique_ptr<mlir::ModuleOp> MLIRGenEntry(mlir::MLIRContext &context, Program& prg) {
        MLIRGen gen = MLIRGen(context);
        mlir::ModuleOp v = gen.mlirGen(prg);
        
        std::unique_ptr<mlir::ModuleOp> val(new mlir::ModuleOp(std::move(v)));
        
        return val;
    }
}
